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
using FanucPCDK;

namespace FanucController
{
    public partial class MainForm : Form
    {

        #region Fields

        // Directories
        public const string TopDir = @"D:\Fanuc Experiments\vx-test-0305";
        public string ReferenceDir;
        public string OutputDir;
        public string ScriptsDir;
        public string LogDir;

        public string LoggerTestPath = Path.Combine(TopDir, "LoggerTest", "log_test_1.txt");

        // Timer (threadpool)
        public System.Timers.Timer Timer;

        // Stopwatch
        public Stopwatch StopWatch;
        public double Time;

        // Form timer
        public System.Windows.Forms.Timer FormTimer;

        // Log Memory
        public MemorySink LogMemory;

        // Vx Path
        public string[] VxPaths;

        // VxApi
        public VXelementsUtility Vx;

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

        #endregion

        #region Constructor
        public MainForm()
        {
            InitializeComponent();

            // Directories
            ReferenceDir = Path.Combine(TopDir, "Reference");
            OutputDir = Path.Combine(TopDir, "Output");
            ScriptsDir = Path.Combine(TopDir, "Scripts");
            LogDir = Path.Combine(TopDir, "Log");
            Directory.CreateDirectory(ReferenceDir);
            Directory.CreateDirectory(OutputDir);
            Directory.CreateDirectory(ScriptsDir);
            Directory.CreateDirectory(LogDir);

            // Vx Paths
            VxPaths = new string[3];
            VxPaths[0] = Path.Combine(OutputDir, "VxRaw.csv");
            VxPaths[1] = Path.Combine(OutputDir, "VxKf.csv");
            VxPaths[2] = Path.Combine(OutputDir, "VxRkf.csv");

            // PCDK
            IsRunning = false;
            IsAborterd = true;
            PoseWFBuffer = new Buffer<PoseData>(20000);
            PoseUFBuffer = new Buffer<PoseData>(20000);
            JointBuffer = new Buffer<JointData>(20000);

            // Timer
            Timer = new System.Timers.Timer();
            Timer.Interval = 500; // 80 ms
            Timer.AutoReset = true;
            //Timer.Elapsed += async (s, e) => await TestEventHandlerAsync();
            //this.Timer.Start();

            // Form Timer
            FormTimer = new System.Windows.Forms.Timer();
            FormTimer.Interval = 80; // 200 ms
            FormTimer.Start();

            // Form closing
            this.FormClosing += VxFormClosed;

            // Stop watch
            StopWatch = new Stopwatch();
            StopWatch.Start();
            Time = 0;

            // Initialize VXelement API
            Vx = new VXelementsUtility(StopWatch);

            // Initialize log display
            LogDisplay.Initiate(listViewLogger, 7);

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

            // Initialize form text
            buttonPcdkConnect.Text = "Connect";
            textBoxPcdkIpAddress.Text = "192.168.0.10";
            textBoxPcdkOffsetX.Text = "0.0";
            textBoxPcdkOffsetY.Text = "0.0";
            textBoxPcdkOffsetZ.Text = "0.0";
            textBoxPcdkOffsetW.Text = "0.0";
            textBoxPcdkOffsetP.Text = "0.0";
            textBoxPcdkOffsetR.Text = "0.0";
        }

        //public async Task TestEventHandlerAsync()
        //{
        //    await Task.Delay(3000);
        //    Console.WriteLine("Hello");
        //}
        #endregion

        #region General


        #endregion

        #region Fanuc Experiment

        private void buttonRunMain_Click(object sender, EventArgs e)
        {

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

        // -------------------------- Helpers

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

        #region Vxelements

        public void VxFormClosed(object sender, EventArgs e)
        {
            Vx.ExitApi();
        }

        private void buttonVxTest_Click(object sender, EventArgs e)
        {
            Vx.TestAction();
        }

        private void buttonVxQuickConnect_Click(object sender, EventArgs e)
        {
            string targetsPath = Path.Combine(ReferenceDir, "targets_0305.txt");
            string modelPath = Path.Combine(ReferenceDir, "test_model_0305.txt");
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
        }

        private void buttonVxStopTracking_Click(object sender, EventArgs e)
        {
            Vx.StopTracking();
            if (checkBoxVxAppend.Checked)
            {
                FormTimer.Tick -= trackingEventHandler;
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

        private void trackingEventHandler(object s, EventArgs e)
        {
            Vx.ExportBuffersLast(VxPaths);
        }

        #endregion

        #region Gui Update

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
                textBoxVxAlpha.Text = (pose[3]*180/Math.PI).ToString("0.00000");
                textBoxVxBeta.Text = (pose[4]*180/Math.PI).ToString("0.00000");
                textBoxVxGamma.Text = (pose[5]*180/Math.PI).ToString("0.00000");
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

        #region Logger

        #endregion

        #region Test
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
                textBoxDebug.AppendText(str + Environment.NewLine);
            }
        }

        private void printToDebug(string str)
        {
            textBoxDebug.AppendText(str + Environment.NewLine);
        }

        #endregion

        #region Junk

        private void listViewLogger_SelectedIndexChanged(object sender, EventArgs e)
        {

        }


        #endregion


    }



}
