using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.Threading;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using Serilog.Configuration;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML.OnnxRuntime;
using FanucPCDK;

namespace FanucController
{
    public partial class MainForm : Form
    {

        #region Fields

        // Directories
        public string MetaTopDir = @"D:\Fanuc Experiments";
        public string TopDir;
        public string ReferenceDir;
        public string OutputDir;
        public string ScriptsDir;
        public string LogDir;

        // Timer (threadpool)
        public System.Timers.Timer Timer;

        // Stopwatch
        public Stopwatch StopWatch;
        public double Time;

        // Form timer
        public System.Windows.Forms.Timer FormTimer;

        // Log Memory
        public MemorySink LogMemory;

        // VxApi
        public VXelementsUtility Vx;

        // Vx Path
        public string[] VxPaths;

        // Vx Pose
        public double[] VxPoseRaw;
        public double[] VxPoseKf;
        public double[] VxPoseRkf;

        // PCDK pose
        public double[] PoseWF;
        public double[] PoseUF;
        public double[] Joint;

        // PCDK Buffers
        public Buffer<PoseData> PoseWFBuffer;
        public Buffer<PoseData> PoseUFBuffer;
        public Buffer<JointData> JointBuffer;

        // PCDK Program Monitoring Bool
        public bool IsRunning;
        public bool IsAborterd;

        // PCDK Paths
        public string[] PcdkPaths;

        // Linear Path Tracking
        public LinearPathTracking LinearTrack;

        // Linear Path Tracking Pose
        public double[] LinearTrackPose;
        public double[] LinearTrackPathError;

        // DataGridView Rows
        public List<DataGridViewRow> DataGridViewRows;
        public Dictionary<string, double[]> ObsDict;

        #endregion

        #region Constructor
        public MainForm()
        {
            InitializeComponent();

            // Directories
            // TopDir = Path.Combine(MetaTopDir, textBoxTopDir.Text);
            TopDir = @textBoxTopDir.Text;
            ReferenceDir = Path.Combine(TopDir, "reference");
            OutputDir = Path.Combine(TopDir, "output");
            // ScriptsDir = Path.Combine(TopDir, "Scripts");
            LogDir = Path.Combine(TopDir, "log");
            Directory.CreateDirectory(TopDir);
            Directory.CreateDirectory(ReferenceDir);
            Directory.CreateDirectory(OutputDir);
            // Directory.CreateDirectory(ScriptsDir);
            Directory.CreateDirectory(LogDir);

            // Initialize log display
            LogDisplay.Initiate(listViewLogger, 2000);

            // Initialize global logger
            string logPath = Path.Combine(LogDir, $"log_{DateTime.Now}.txt");
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Memory()
                .WriteTo.Console()
                .WriteTo.File(logPath, rollingInterval: RollingInterval.Day)
                .CreateLogger();
            Log.Information("Global logger configured!");
            LogMemory = MemorySink.Instance;

            // Timer
            Timer = new System.Timers.Timer();
            Timer.Interval = 80; // 80 ms
            Timer.AutoReset = true;
            Timer.Start();
            Log.Information("System timer started");

            // Form Timer
            FormTimer = new System.Windows.Forms.Timer();
            FormTimer.Interval = 150; // 150 ms
            FormTimer.Start();
            Log.Information("Form timer started");

            // Stop watch
            StopWatch = new Stopwatch();
            StopWatch.Start();
            Time = 0;

            // Form closing
            this.FormClosing += VxFormClosed;

            // Vx Paths
            VxPaths = new string[3];
            VxPaths[0] = Path.Combine(OutputDir, "VxRaw.csv");
            VxPaths[1] = Path.Combine(OutputDir, "VxKf.csv");
            VxPaths[2] = Path.Combine(OutputDir, "VxRkf.csv");
            // Initialize VXelement API
            Vx = new VXelementsUtility(StopWatch);
            VxPoseRaw = new double[6]; VxPoseKf = new double[6]; VxPoseRkf = new double[6];

            // PCDK
            IsRunning = false;
            IsAborterd = true;
            PoseWFBuffer = new Buffer<PoseData>(20000);
            PoseUFBuffer = new Buffer<PoseData>(20000);
            JointBuffer = new Buffer<JointData>(20000);
            PoseWF = new double[6]; PoseUF = new double[6]; Joint = new double[6];

            // Initialize PCDK form text
            buttonPcdkConnect.Text = "Connect";
            textBoxPcdkIpAddress.Text = "192.168.0.10";
            textBoxPcdkOffsetX.Text = "0.0";
            textBoxPcdkOffsetY.Text = "0.0";
            textBoxPcdkOffsetZ.Text = "0.0";
            textBoxPcdkOffsetW.Text = "0.0";
            textBoxPcdkOffsetP.Text = "0.0";
            textBoxPcdkOffsetR.Text = "0.0";

            // Initialize Linear Path Tracking
            LinearTrack = new LinearPathTracking(Vx, Timer, StopWatch, TopDir);
            LinearTrackPose = new double[6]; LinearTrackPathError = new double[6];

            // Initialize 
            DataGridViewRows = new List<DataGridViewRow>();
            ObsDict = new Dictionary<string, double[]>();

        }

        #endregion

        #region General


        #endregion

        #region Linear Path Tracking

        private void buttonLineTrackQuickSetup_Click(object sender, EventArgs e)
        {
            // Data Grid View
            buttonDgvShow_Click(sender, e);
            buttonVxDisplay_Click(sender, e);
            buttonPcdkDisplay_Click(sender, e);
            buttonDisplay_Click(sender, e);

            // PCDK
            buttonPcdkConnect_Click(sender, e);
            buttonPcdkAttachAll_Click(sender, e);

            // Vx
            buttonVxQuickConnect_Click(sender, e);
            buttonVxStartTracking_Click(sender, e);

            // Linear Tracking
            buttonLinearPathLoadJson_Click(sender, e);
            buttonRotationIdLoadJson_Click(sender, e);
            buttonPoseAttach_Click(sender, e);

            // Control
            checkBoxLineTrackStep.Checked = false;
            checkBoxLineTrackDPM.Checked = false;
            checkBoxLineTrackPControl.Checked = true;
            checkBoxLineTrackIlc.Checked = false;  
            checkBoxLineTrackPNN.Checked = true;
            checkBoxLineTrackMBPO.Checked = false;
            checkBoxLineTrackBPNNPID.Checked = false;
            textBoxLineTrackIter.Text = "1";
        }

        private void buttonRunMain_Click(object sender, EventArgs e)
        {
            string progName = textBoxLineTrackProgName.Text;
            string pathPath = Path.Combine(ReferenceDir, "LinearPath.json");
            int iter = int.Parse(textBoxLineTrackIter.Text);
            bool dpm = checkBoxLineTrackDPM.Checked;
            bool pControl = checkBoxLineTrackPControl.Checked;
            bool ilc = checkBoxLineTrackIlc.Checked;
            bool pNN = checkBoxLineTrackPNN.Checked;
            bool mbpo = checkBoxLineTrackMBPO.Checked;
            bool bpnnpid = checkBoxLineTrackBPNNPID.Checked;
            bool step = checkBoxLineTrackStep.Checked;
            LinearTrack.Init(pathPath, progName, iter, dpm:dpm, pControl:pControl,
                ilc:ilc, pNN:pNN, mbpo:mbpo, bpnnpid:bpnnpid,step:step);
            LinearTrack.Start();
        }

        private void buttonRecordPose_Click(object sender, EventArgs e)
        {
            string key = comboBoxPoseDict.Text;
            int sampleNum = 20;
            Vector<double> pose = CreateVector.Dense<double>(6);
            switch (comboBoxPoseFrame.Text)
            {
                case "World Frame":
                    pose = LinearTrack.GetVxCameraPose(sampleNum);
                    break;

                case "User Frame":
                    pose = LinearTrack.GetVxUFPose(sampleNum);
                    break;
            }
            LinearTrack.RecordPose(key, pose);
        }

        private void buttonRotationId_Click(object sender, EventArgs e)
        {
            LinearTrack.RotationIdentify(comboBoxRotationXyz.Text);
        }

        private void buttonRotationIdWriteJson_Click(object sender, EventArgs e)
        {
            LinearTrack.RotationId.ToJson(ReferenceDir);
        }

        private void buttonRotationIdLoadJson_Click(object sender, EventArgs e)
        {
            LinearTrack.RotationId.ReadJson(ReferenceDir);
        }

        private void buttonLinearPath_Click(object sender, EventArgs e)
        {
            LinearTrack.LinearPath.GetData(LinearTrack.PoseDict, LinearTrack.RotationId);
            LinearTrack.LinearPath.fileName = "LinearPath.json";
            LinearTrack.LinearPath.ToJson(ReferenceDir);
        }

        private void buttonLinearPathLoadJson_Click(object sender, EventArgs e)
        {
            LinearTrack.LinearPath.ReadJson(ReferenceDir);
            Log.Information($"Loaded linear path from {Path.GetDirectoryName(ReferenceDir)} directory");
        }

        private void buttonPoseAttach_Click(object sender, EventArgs e)
        {
            if (buttonPoseAttach.Text == "Attach")
            {
                buttonPoseAttach.Enabled = false;
                Timer.Elapsed += buttonPoseGet_Click;
                buttonPoseAttach.Text = "Detach";
                buttonPoseAttach.Enabled = true;
            }
            else if (buttonPoseAttach.Text == "Detach")
            {
                buttonPoseAttach.Enabled = false;
                Timer.Elapsed -= buttonPoseGet_Click;
                buttonPoseAttach.Text = "Attach";
                buttonPoseAttach.Enabled = true;
            }
        }

        private void buttonDisplay_Click(object sender, EventArgs e)
        {
            if (buttonDisplay.Text == "Display")
            {
                buttonDisplay.Enabled = false;
                addDataGridViewRow("LinearTrack", "LinearTrackPose");
                addDataGridViewRow("LinearTrack", "LinearTrackPathError");
                buttonDisplay.Text = "Hide";
                buttonDisplay.Enabled = true;
            }
            else if (buttonDisplay.Text == "Hide")
            {
                buttonDisplay.Enabled = false;
                removeDataGridViewRow("LinearTrack");
                removeDataGridViewRow("LinearTrack");
                buttonDisplay.Text = "Display";
                buttonDisplay.Enabled = true;
            }
        }

        private void buttonPoseGet_Click(object sender, EventArgs e)
        {
            updateLinearTrackPose();
            updateLinearTrackTxtBox();
        }

        #region Linear Track Helpers

        private void updateLinearTrackPose()
        {
            LinearTrackPose = LinearTrack.GetVxUFPose().AsArray();
            LinearTrackPathError = LinearTrack.PathError.AsArray();
        }


        #endregion

        #endregion

        #region Linear Path ILC Tracking

        private void buttonIlcRun_Click(object sender, EventArgs e)
        {
            string pathPath = Path.Combine(ReferenceDir, "LinearPath.json");
            LinearTrack.Init(pathPath, "TEST1214", 1, dpm: false);
            LinearTrack.Start();
        }

        #endregion

        #region PCDK

        private void buttonPcdkConnect_Click(object sender, EventArgs e)
        {
            if (buttonPcdkConnect.Text == "Connect")
            {
                buttonPcdkConnect.Enabled = false;
                PCDK.Connect(textBoxPcdkIpAddress.Text);
                PCDK.SetReferenceFrame(); // Should be apart of connect
                PCDK.SetupJoint();
                PCDK.SetupIO();
                textBoxPcdkRobotName.Text = PCDK.RobotName;
                buttonPcdkConnect.Text = "Disconnect";
                buttonPcdkConnect.Enabled = true;
                Log.Information("Connected to FANUC Controller!");
            }
            else if (buttonPcdkConnect.Text == "Disconect")
            {
                buttonPcdkConnect.Enabled = false;
                PCDK.Disconnect();
                textBoxPcdkRobotName.Text = "";
                buttonPcdkConnect.Text = "Connected";
                buttonPcdkConnect.Enabled = true;
                Log.Information("Disconnected from FANUC Controller!");
            }
        }

        private void buttonPcdkGetPose_Click(object sender, EventArgs e)
        {
            getPcdkPose();
            if (checkBoxPcdkDisplay.Checked)
            {
                updatePCDKTxtBox(PoseWF, PoseUF, null);
            }
            if (checkBoxPcdkPoseAppend.Checked)
            {
                string path;
                path = Path.Combine(OutputDir, "PcdkPoseWF.csv");
                appendPcdkPoseCsv(path, PoseWFBuffer);
                path = Path.Combine(OutputDir, "PcdkPoseUF.csv");
                appendPcdkPoseCsv(path, PoseUFBuffer);
            }
        }

        private void buttonAttachPose_Click(object sender, EventArgs e)
        {
            if (buttonPcdkAttachPose.Text == "Attach")
            {
                buttonPcdkAttachPose.Enabled = false;
                FormTimer.Tick += buttonPcdkGetPose_Click;
                addDataGridViewRow("Pcdk");
                if (checkBoxPcdkPoseAppend.Checked)
                {
                    string path;
                    path = Path.Combine(OutputDir, "PcdkPoseWF.csv");
                    Csv.AppendCsv(path, new PoseData(new double[6], 0), append: false, header: true);
                    path = Path.Combine(OutputDir, "PcdkPoseUF.csv");
                    Csv.AppendCsv(path, new PoseData(new double[6], 0), append: false, header: true);
                }
                buttonPcdkAttachPose.Text = "Detach";
                buttonPcdkAttachPose.Enabled = true;
            }
            else if (buttonPcdkAttachPose.Text == "Detach")
            {
                buttonPcdkAttachPose.Enabled = false;
                FormTimer.Tick -= buttonPcdkGetPose_Click;
                buttonPcdkAttachPose.Text = "Attach";
                buttonPcdkAttachPose.Enabled = true;
            }

        }

        private void buttonPcdkExportPose_Click(object sender, EventArgs e)
        {
            List<PoseData> poseData;
            string path;
            path = Path.Combine(OutputDir, "PcdkPoseWF.csv");
            poseData = PoseWFBuffer.Memory.ToList();
            Csv.WriteCsv(path, poseData);
            path = Path.Combine(OutputDir, "PcdkPoseUF.csv");
            poseData = PoseUFBuffer.Memory.ToList();
            Csv.WriteCsv(path, poseData);
        }

        private void buttonPcdkGetJoint_Click(object sender, EventArgs e)
        {
            getPcdkJoint();
            if (checkBoxPcdkDisplay.Checked)
            {
                updatePCDKTxtBox(null, null, Joint);
            }
            if (checkBoxPcdkJointAppend.Checked)
            {
                string path;
                path = Path.Combine(OutputDir, "PcdkJoint.csv");
                appendPcdkJointCsv(path, JointBuffer);
            }
        }

        private void buttonPcdkAttachJoint_Click(object sender, EventArgs e)
        {
            if (buttonPcdkAttachJoint.Text == "Attach")
            {
                buttonPcdkAttachJoint.Enabled = false;
                FormTimer.Tick += buttonPcdkGetJoint_Click;
                if (checkBoxPcdkJointAppend.Checked)
                {
                    string path;
                    path = Path.Combine(OutputDir, "PcdkJoint.csv");
                    Csv.AppendCsv(path, new JointData(new double[6], 0), append: false, header: true);
                }
                buttonPcdkAttachJoint.Text = "Detach";
                buttonPcdkAttachJoint.Enabled = true;
            }
            else if (buttonPcdkAttachJoint.Text == "Detach")
            {
                buttonPcdkAttachJoint.Enabled = false;
                FormTimer.Tick -= buttonPcdkGetJoint_Click;
                buttonPcdkAttachJoint.Text = "Attach";
                buttonPcdkAttachJoint.Enabled = true;
            }
        }

        private void buttonPcdkAttachAll_Click(object sender, EventArgs e)
        {
            buttonPcdkAttachJoint_Click(sender, e);
            buttonAttachPose_Click(sender, e);
        }

        private void buttonPcdkExportJoint_Click(object sender, EventArgs e)
        {
            string path = Path.Combine(OutputDir, "PcdkJoint.csv");
            List<JointData> jointData = JointBuffer.Memory.ToList();
            Csv.WriteCsv(path, jointData);
        }

        private void buttonDpmOffset_Click(object sender, EventArgs e)
        {
            double[] u = new double[6];
            u[0] = Convert.ToDouble(textBoxPcdkOffsetX.Text);
            u[1] = Convert.ToDouble(textBoxPcdkOffsetY.Text);
            u[2] = Convert.ToDouble(textBoxPcdkOffsetZ.Text);
            u[3] = Convert.ToDouble(textBoxPcdkOffsetW.Text);
            u[4] = Convert.ToDouble(textBoxPcdkOffsetP.Text);
            u[5] = Convert.ToDouble(textBoxPcdkOffsetR.Text);
            if (u.Select(x => x * x).Sum() < 5 * 5)
            {
                //PCDK.SetupDPM(sch: (int)numericUpDownPcdkDpmSch.Value);
                PCDK.ApplyDPM(u, sch: (int)numericUpDownPcdkDpmSch.Value);
            }
        }

        private void buttonPcdkDisplay_Click(object sender, EventArgs e)
        {
            if (buttonPcdkDisplay.Text == "Display")
            {
                buttonPcdkDisplay.Enabled = false;
                addDataGridViewRow("Pcdk", "PcdkPoseUF");
                addDataGridViewRow("Pcdk", "PcdkPoseWF");
                addDataGridViewRow("Pcdk", "PcdkJoint");
                buttonPcdkDisplay.Text = "Hide";
                buttonPcdkDisplay.Enabled = true;
            }
            else if (buttonPcdkDisplay.Text == "Hide")
            {
                buttonPcdkDisplay.Enabled = false;
                removeDataGridViewRow("Pcdk");
                removeDataGridViewRow("Pcdk");
                removeDataGridViewRow("Pcdk");
                buttonPcdkDisplay.Text = "Display";
                buttonPcdkDisplay.Enabled = true;
            }
        }

        #region PCDK Helpers
        private void monitorPcdkProgram(string programName)
        {
            IsRunning = PCDK.IsRunning();
            IsAborterd = PCDK.IsAborted();
        }

        private void appendPcdkPoseCsv(string path, Buffer<PoseData> poseBuffer)
        {
            PoseData poseData;
            poseData = poseBuffer.Memory.Last();
            Csv.AppendCsv(path, poseData, append: true, header: false);
        }

        private void appendPcdkJointCsv(string path, Buffer<JointData> jointBuffer)
        {
            JointData jointData;
            jointData = jointBuffer.Memory.Last();
            Csv.AppendCsv(path, jointData, append: true, header: false);
        }

        private void getPcdkPose()
        {
            Time = StopWatch.Elapsed.TotalSeconds;
            PCDK.SetReferenceFrame();
            PoseWF = PCDK.GetPoseWF();
            PoseUF = PCDK.GetPoseUF();
            PoseWFBuffer.Add(new PoseData(PoseWF, Time));
            PoseUFBuffer.Add(new PoseData(PoseUF, Time));
        }

        private void getPcdkJoint()
        {
            Time = StopWatch.Elapsed.TotalSeconds;
            Joint = PCDK.GetJointPosition();
            JointBuffer.Add(new JointData(Joint, Time));
        }

        #endregion

        #endregion

        #region Vxelements

        private void buttonVxTest_Click(object sender, EventArgs e)
        {
            Vx.TestAction();
        }

        private void buttonVxQuickConnect_Click(object sender, EventArgs e)
        {
            string targetsPath = Path.Combine(ReferenceDir, "targets_0510.txt");
            string modelPath = Path.Combine(ReferenceDir, "model_0510.txt");
            Vx.QuickConnect(targetsPath, modelPath);
            // Activate the necessary filters
            Vx.filterActivated[0] = checkBoxVxRaw.Checked;
            Vx.filterActivated[1] = checkBoxVxKf.Checked;
            Vx.filterActivated[2] = checkBoxVxRkf.Checked;
        }

        private void buttonVxStartTracking_Click(object sender, EventArgs e)
        {

            Vx.StartTracking();
            Vx.ExportBuffersLast(VxPaths, append: false, header: true);
            if (checkBoxVxAppend.Checked)
            {
                FormTimer.Tick += trackingEventHandler;
            }
            FormTimer.Tick += (s, ev) => updateVxTxtBox();
            FormTimer.Tick += updateVxPose;
        }

        private void buttonVxStopTracking_Click(object sender, EventArgs e)
        {
            Vx.StopTracking();
            if (checkBoxVxAppend.Checked)
            {
                FormTimer.Tick -= trackingEventHandler;
                FormTimer.Tick -= updateVxPose;
            }
        }

        private void buttonVxDisplay_Click(object sender, EventArgs e)
        {
            if (buttonVxDisplay.Text == "Display")
            {
                buttonVxDisplay.Enabled = false;
                addDataGridViewRow("Vx", "VxPoseRaw");
                addDataGridViewRow("Vx", "VxPoseKf");
                addDataGridViewRow("Vx", "VxPoseRkf");
                buttonVxDisplay.Text = "Hide";
                buttonVxDisplay.Enabled = true;
            }
            else if (buttonVxDisplay.Text == "Hide")
            {
                buttonVxDisplay.Enabled = false;
                removeDataGridViewRow("Vx");
                removeDataGridViewRow("Vx");
                removeDataGridViewRow("Vx");
                buttonVxDisplay.Text = "Display";
                buttonVxDisplay.Enabled = true;
            }
            
        }

        private void buttonVxReset_Click(object sender, EventArgs e)
        {
            Vx.Reset();
        }

        private void buttonVxExit_Click(object sender, EventArgs e)
        {
            Vx.Exit();
        }

        private void buttonVxExport_Click(object sender, EventArgs e)
        {
            Vx.ExportBuffers(VxPaths);
        }

        #region Vx Helpers

        public void VxFormClosed(object sender, EventArgs e)
        {
            Vx.ExitApi();
        }

        private void trackingEventHandler(object s, EventArgs e)
        {
            Vx.ExportBuffersLast(VxPaths);
        }

        private void updateVxPose(object s, EventArgs e)
        {
            VxPoseRaw = Vx.PoseCameraFrame.AsArray();
            VxPoseKf = Vx.PoseCameraFrameKf.AsArray();
            VxPoseRkf = Vx.PoseCameraFrameRkf.AsArray();
        }

        #endregion

        #endregion

        #region Gui Update

        public void updateLinearTrackTxtBox()
        {
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updateLinearTrackTxtBox();
                this.Invoke(safeUpdate);
            }
            else
            {
                Vector<double> pose;
                pose = LinearTrack.GetVxUFPose();
                textBoxLinearPathX.Text = pose[0].ToString();
                textBoxLinearPathY.Text = pose[1].ToString();
                textBoxLinearPathZ.Text = pose[2].ToString();
                textBoxLinearPathW.Text = pose[3].ToString();
                textBoxLinearPathP.Text = pose[4].ToString();
                textBoxLinearPathR.Text = pose[5].ToString();
            }
        }

        public void updateVxTxtBox()
        {
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updateVxTxtBox();
                this.Invoke(safeUpdate);
            }
            else
            {
                Vector<double> pose;
                pose = Vx.PoseCameraFrame;
                switch (comboBoxVxFilter.Text)
                {
                    case "RAW":
                        break;
                    case "KF":
                        pose = Vx.PoseCameraFrameKf;
                        break;
                    case "RKF":
                        pose = Vx.PoseCameraFrameRkf;
                        break;
                }

                textBoxVxX.Text = pose[0].ToString("0.00000");
                textBoxVxY.Text = pose[1].ToString("0.00000");
                textBoxVxZ.Text = pose[2].ToString("0.00000");
                textBoxVxAlpha.Text = (pose[3] * 180 / Math.PI).ToString("0.00000");
                textBoxVxBeta.Text = (pose[4] * 180 / Math.PI).ToString("0.00000");
                textBoxVxGamma.Text = (pose[5] * 180 / Math.PI).ToString("0.00000");
            }
        }

        private void updatePCDKTxtBox(double[] poseWF, double[] poseUF, double[] joint)
        {
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updatePCDKTxtBox(poseWF, poseUF, joint);
                this.Invoke(safeUpdate);
            }
            else
            {
                double[] pose = null;
                switch (comboBoxPcdkPose.Text)
                {
                    case "World Frame":
                        pose = poseWF;
                        break;
                    case "User Frame":
                        pose = poseUF;
                        break;
                }

                if (pose != null)
                {
                    textBoxPcdkX.Text = pose[0].ToString();
                    textBoxPcdkY.Text = pose[1].ToString();
                    textBoxPcdkZ.Text = pose[2].ToString();
                    textBoxPcdkW.Text = pose[3].ToString();
                    textBoxPcdkP.Text = pose[4].ToString();
                    textBoxPcdkR.Text = pose[5].ToString();
                }
                if (joint != null)
                {
                    textBoxPcdkJ1.Text = joint[0].ToString();
                    textBoxPcdkJ2.Text = joint[1].ToString();
                    textBoxPcdkJ3.Text = joint[2].ToString();
                    textBoxPcdkJ4.Text = joint[3].ToString();
                    textBoxPcdkJ5.Text = joint[4].ToString();
                    textBoxPcdkJ6.Text = joint[5].ToString();
                }
            }
        }

        #endregion

        #region DataGridView

        private void buttonDgvClear_Click(object sender, EventArgs e)
        {
            foreach (DataGridViewRow tempRow in dataGridView.Rows)
            {
                try
                {
                    dataGridView.Rows.Remove(tempRow);
                    buttonDgvClear_Click(sender, e);
                }
                catch (Exception ex)
                {
                    Log.Error(ex.ToString());
                }
            }
        }

        private void buttonDgvShow_Click(object sender, EventArgs e)
        {
            if (buttonDgvShow.Text == "Show")
            {
                buttonDgvShow.Enabled = false;
                FormTimer.Tick += updateObsDict;
                FormTimer.Tick += (s, ev) => updateDataGridViewRows(ObsDict);
                buttonDgvShow.Text = "Hide";
                buttonDgvShow.Enabled = true;
            }
            else if (buttonDgvShow.Text == "Hide")
            {
                buttonDgvShow.Enabled = false;
                FormTimer.Tick -= updateObsDict;
                FormTimer.Tick -= (s, ev) => updateDataGridViewRows(ObsDict);
                buttonDgvShow.Text = "Show";
                buttonDgvShow.Enabled = true;
            }
        }

        #region Dgv Helpers

        private void updateObsDict(object s, EventArgs e)
        {
            ObsDict["PcdkPoseUF"] = PoseUF;
            ObsDict["PcdkPoseWF"] = PoseWF;
            ObsDict["PcdkJoint"] = Joint;
            ObsDict["VxPoseRaw"] = VxPoseRaw;
            ObsDict["VxPoseKf"] = VxPoseKf;
            ObsDict["VxPoseRkf"] = VxPoseRkf;
            ObsDict["LinearTrackPose"] = LinearTrackPose;
            ObsDict["LinearTrackPathError"] = LinearTrackPathError;
        }

        private void addDataGridViewRow(string header, string value = null)
        {
            DataGridViewComboBoxCell comboHeaderCell = new DataGridViewComboBoxCell();
            DataGridViewRow headerRow = new DataGridViewRow();
            DataGridViewRow row = new DataGridViewRow();
            switch (header)
            {
                case "Pcdk":
                    headerRow.CreateCells(dataGridView, header, "X", "Y", "Z", "W", "P", "R");
                    comboHeaderCell.DataSource = new object[] { "PcdkPoseUF", "PcdkPoseWF", "PcdkJoint" };
                    if (String.IsNullOrEmpty(value)) { value = "PcdkPoseUF"; }
                    comboHeaderCell.Value = value;
                    row.CreateCells(dataGridView);
                    row.Cells[0] = comboHeaderCell;
                    break;
                case "Vx":
                    headerRow.CreateCells(dataGridView, header, "X", "Y", "Z", "W", "P", "R");
                    comboHeaderCell.DataSource = new object[] { "VxPoseRaw", "VxPoseKf", "VxPoseRkf" };
                    if (String.IsNullOrEmpty(value)) { value = "VxPoseRKF"; }
                    comboHeaderCell.Value = value;
                    row.CreateCells(dataGridView);
                    row.Cells[0] = comboHeaderCell;
                    break;
                case "LinearTrack":
                    headerRow.CreateCells(dataGridView, header, "X", "Y", "Z", "W", "P", "R");
                    comboHeaderCell.DataSource = new object[] { "LinearTrackPose", "LinearTrackPathError" };
                    if (String.IsNullOrEmpty(value)) { value = "LinearTrackPose"; }
                    comboHeaderCell.Value = value;
                    row.CreateCells(dataGridView);
                    row.Cells[0] = comboHeaderCell;
                    break;
            }
            dataGridView.Rows.Add(headerRow);
            dataGridView.Rows.Add(row);
        }

        private void removeDataGridViewRow(string header)
        {
            // Remove row based on the text of the first cell in each row
            DataGridViewRow headerRow = new DataGridViewRow();
            DataGridViewRow row = new DataGridViewRow();
            foreach (DataGridViewRow tempRow in dataGridView.Rows)
            {
                if (tempRow.Cells[0].Value == header)
                {
                    int idx = tempRow.Index;
                    headerRow = dataGridView.Rows[idx];
                    row = dataGridView.Rows[idx + 1];
                }
            }
            try
            {
                dataGridView.Rows.Remove(headerRow);
                dataGridView.Rows.Remove(row);
            }
            catch (Exception ex)
            {
                Log.Error(ex.ToString());
            }
        }

        private void updateDataGridViewRows(Dictionary<string, double[]> obsDict)
        {
            // Update DataGridView based on the text of the first cell in each row
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updateDataGridViewRows(obsDict);
                this.Invoke(safeUpdate);
            }
            else
            {
                foreach (DataGridViewRow row in dataGridView.Rows)
                {
                    string header;
                    switch (row.Cells[0].Value)
                    {
                        case "PcdkPoseUF":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "PcdkPoseWF":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "PcdkJoint":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "VxPoseRkf":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < 6; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "VxPoseKf":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < 6; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "VxPoseRaw":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "LinearTrackPose":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                        case "LinearTrackPathError":
                            header = row.Cells[0].Value.ToString();
                            for (int i = 0; i < obsDict[header].Length; i++)
                            {
                                row.Cells[i + 1].Value = obsDict[header][i].ToString("0.00000");
                            }
                            break;
                    }
                }
            }

            #endregion

        }

        #endregion

        #region Logger

        #endregion

        #region Test

        private void buttonTestTest_Click(object sender, EventArgs e)
        {
            var kp = CreateMatrix.Dense<double>(6, 6);
            kp[0, 0] = 0.03; kp[1, 1] = 0.03; kp[2, 2] = 0.03;
            var kd = CreateMatrix.Dense<double>(6, 6);
            var ilc = new IterativeLearningControl(kp, kd);
            var ilc_last = new IterativeLearningControl(kp, kd);
            string controlPath = @"D:\fanuc experiments\test-0327-test\output\iteration_0\LineTrackIlcControl.csv";
            string errorPath = @"D:\fanuc experiments\test-0327-test\output\iteration_0\LineTrackError.csv";
            string saveIlcPath = @"D:\fanuc experiments\test-0327-test\output\iteration_0\LineTrackIlcNextControl.csv";
            ilc.FromCsv(controlPath, errorPath);
            ilc.PControl();
            ilc.ToCsv(saveIlcPath);

            string dataDir = @"D:\fanuc experiments\test-0327-test\output\iteration_0";
            string savePath = @"D:\Fanuc Experiments\test-0327-test\output\iteration_0\test_fig";
            string[] args = new string[2] { dataDir, savePath };
            // args = new string[1] { dataDir };
            PythonScripts.RunParallel(scriptName: "iter_plot.py", args: args);

            savePath = @"D:\Fanuc Experiments\test-0327-test\output\iteration_0\test_ilc_fig";
            args = new string[2] { dataDir, savePath };
            PythonScripts.RunParallel(scriptName: "iter_ilc_plot.py", args: args);


        }

        private void buttonPnnTest_Click(object sender, EventArgs e)
        {
            //LinearPathTrackingPNN.Test();
            var pnn = new LinearPathTrackingPNN();

            List<string> dataDirs = new List<string>()
            {
                @"D:\Fanuc Experiments\pcontrol-test-0416\output"
            };
            pnn.Iteration(nEpoch: 50, dataDirs: dataDirs);
            pnn.NewSession();
            pnn.Plot(@"D:\Fanuc Experiments\pcontrol-test-0416\output");

            var error = pnn.Forward(new double[] { 0.0, 0.05, 0.05, 0.05 });
            Console.WriteLine($"Error: {string.Join(",", error)}");
        }

        private void buttonMbpoTest_Click(object sender, EventArgs e)
        {
            string iterDir = @"D:\Fanuc Experiments\test-master\output\iteration_0";

            var mbpo = new LinearPathTrackingMBPO
            {
                OutputDir = @"D:\Fanuc Experiments\test-master\output",
                WarmupIters = 0
            };
            List<string> dataDirs = new List<string>()
            {
                @"D:\Fanuc Experiments\test-master\output"
            };
            mbpo.Init();
            mbpo.Reset();

            // Control
            Vector<double> error = CreateVector.Dense<double>(6);
            double time = 5;
            var control = mbpo.Control(error, error);

            mbpo.Iteration(nEpoch: 10, gradSteps:500, iterDir, dataDirs:dataDirs);
            mbpo.Plot(iterDir: iterDir);
        }

        private void buttonBpnnpidTest_Click(object sender, EventArgs e)
        {
            var bpnnpid = new LinearPathTrackingBPNNPID();
            Vector<double> reference = CreateVector.DenseOfArray(new double[] { -1953, -1080, -74 });
            Vector<double> actual = CreateVector.DenseOfArray(new double[] { -1800, -900, -60 });
            Vector<double> error = CreateVector.DenseOfArray(new double[] { -0.05, 0.1, 0.3 });
            var control = bpnnpid.Control(reference, actual, error);
            Console.WriteLine($"BPNNPID control: {string.Join(",", control.AsArray())}");
        }

        private void buttonOUTest_Click(object sender, EventArgs e)
        {
            var ou = new OrnsteinUlhenbeckNoise(new double[3] { 0.002, 0.002, 0.002 },
                new double[3] { 0.2, 0.2, 0.2 });
            var x = ou.Sample();
        }

        #endregion

        #region Junk

        private void listViewLogger_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void buttonAddRow_Click(object sender, EventArgs e)
        {
            //dataGridView.RowHeadersWidth = 100;
            //foreach (DataGridViewRow row in dataGridView.Rows)
            //{
            //    if (row.IsNewRow)
            //    {
            //        row.HeaderCell.Value = "PCDK Pose WF";
            //        foreach (DataGridViewCell cell in row.Cells)
            //        {
            //            cell.Value = "2";
            //        }
            //    }
            //}
            //var row1 = new string[6] { "1", "2", "3", "4", "5", "6"};
            //DataGridViewRow x = new DataGridViewRow();
            //x.HeaderCell.Value = "s";
            //dataGridView.Rows.Add(x);
            //var y = dataGridView.Rows;
            ObsDict["PcdkPoseUF"] = new double[] { 1, 1, 1, 1, 1, 1 };
            addDataGridViewRow("Pcdk");
            updateDataGridViewRows(ObsDict);
            //Thread.Sleep(1000);
            //removeDataGridViewRow("Pcdk");

        }


        private void buttonDeleteRow_Click(object sender, EventArgs e)
        {
            removeDataGridViewRow("Pcdk");
        }



        private void buttonTest_Click(object sender, EventArgs e)
        {
            //string level = "Debug";
            //string message = "Testing ListView!";
            //ListViewItem listViewItem = new ListViewItem(level);
            //listViewItem.SubItems.Add(message);
            //listViewItem.SubItems.Add(level);

            //listViewLogger.Items.Add(listViewItem);
            //printToDebug(new string[] { level, message });

            //var logEvent = this.Memory.LogEvents.Last();
            //var logMessage = logEvent.RenderMessage();
            //var logLevel = logEvent.Level.ToString();
            //printToDebug(logLevel + ":" + logMessage);
            //printToDebug(logLevel.ToString());
            //Log.Warning("{Date} Hello", DateTime.Now);
            //var matrix = CreateMatrix.Dense<double>(3, 3);
            //var jsonTest = new JsonTest();
            //jsonTest.Rotation = new double[9];
            //jsonTest.PointA = new double[3];
            //jsonTest.PointB = new double[] { 0, 0, 0 };
            //string jsonString = JsonSerializer.Serialize(jsonTest);
            //Log.Information(jsonString);
            //string jsonPath = Path.Combine(TopDir, "JsonTest", "test.json");
            //File.WriteAllText(jsonPath, jsonString);
            //jsonPath = File.ReadAllText(jsonPath);
            //var jsonTest2 = JsonSerializer.Deserialize<JsonTest>(jsonString);
            //Log.Information(jsonTest2.Rotation[4].ToString());
            var jsonTest = new LinearPath
            {
                Rotation = CreateMatrix.Dense<double>(3, 3),
                PointStart = CreateVector.Dense<double>(3),
                PointEnd = CreateVector.Dense<double>(3)
            };
            string jsonString = JsonSerializer.Serialize(jsonTest);
            Log.Debug(jsonString);
            var jsonTest2 = JsonSerializer.Deserialize<LinearPath>(jsonString);
            Log.Debug(jsonTest2.Rotation.ToString());

        }

        public struct JsonTest
        {
            private Matrix<double> rotation;
            private Vector<double> pointA;
            private Vector<double> pointB;

            public double[] Rotation
            {
                get { return rotation.ToColumnMajorArray(); }
                set { rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
            }

            public double[] PointA
            {
                get { return pointA.ToArray(); }
                set { pointA = CreateVector.DenseOfArray<double>(value); }
            }

            public double[] PointB
            {
                get { return pointB.ToArray(); }
                set { pointB = CreateVector.DenseOfArray<double>(value); }
            }
        }

        private void printToDebug(string[] strArray)
        {
            foreach (string str in strArray)
            {
                //textBoxDebug.AppendText(str + Environment.NewLine);
            }
        }

        private void printToDebug(string str)
        {
            //textBoxDebug.AppendText(str + Environment.NewLine);
        }



        private void dataGridView_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }


        #endregion


    }



}
