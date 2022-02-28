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
        // --------------------------------- Fields
        // Directories
        public const string TopDir = @"D:\.NET Test";
        public string ReferenceDir;
        public string OutputDir;
        public string ScriptsDir;

        public string LoggerTestPath = Path.Combine(TopDir, "LoggerTest", "log_test_1.txt");

        //public System.Timers.Timer Timer;

        // Stopwatch
        public Stopwatch StopWatch;
        public double Time;

        // Form timer
        public System.Windows.Forms.Timer FormTimer;
        
        // VxApi
        public VXelementsUtility Vx;
        
        // Log Memory
        public MemorySink LogMemory;

        // PCDK Buffers
        public Buffer<PoseData> poseBuffer;
        public Buffer<JointData> jointBuffer;

        // --------------------------------- Constructor
        public MainForm()
        {
            InitializeComponent();

            // Directories
            ReferenceDir = Path.Combine(TopDir, "Reference");
            OutputDir = Path.Combine(TopDir, "Output");
            ScriptsDir = Path.Combine(TopDir, "Scripts");
            Directory.CreateDirectory(ReferenceDir);
            Directory.CreateDirectory(OutputDir);
            Directory.CreateDirectory(ScriptsDir);

            // Timer
            //Timer = new System.Timers.Timer();
            //Timer.Interval = 100;
            //Timer.AutoReset = true;
            //Timer.Elapsed += async (s, e) => await TestEventHandlerAsync();
            //this.Timer.Start();

            // Form Timer
            FormTimer = new System.Windows.Forms.Timer();
            FormTimer.Interval = 100;
            FormTimer.Start();

            // Stop watch
            StopWatch = new Stopwatch();
            Time = 0;
            Stopwatch.StartNew();

            // Initialize VXelement API
            Vx = new VXelementsUtility();

            // Initialize log display
            LogDisplay.Initiate(listViewLogger, 7);

            // Initialize global logger
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Memory()
                .WriteTo.Console()
                .WriteTo.File(LoggerTestPath, rollingInterval: RollingInterval.Day)
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

        // --------------------------------- Fanuc Experiment

        // --------------------------------- PCDK

        private void buttonPcdkConnect_Click(object sender, EventArgs e)
        {
            if (buttonPcdkConnect.Text == "Connect")
            {
                buttonPcdkConnect.Enabled = false;
                PCDK.Connect(textBoxPcdkIpAddress.Text);
                PCDK.SetReferenceFrame();
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
            double?[] pose = getPcdkPose();
            Time = StopWatch.ElapsedMilliseconds / 1000;
            addToPcdkBuffers(pose, null, Time);
            updatePCDKTxtBox(pose, null);
        }

        private void buttonAttachPose_Click(object sender, EventArgs e)
        {
            if (buttonPcdkAttachPose.Text == "Attach")
            {
                buttonPcdkAttachPose.Enabled = false;
                FormTimer.Tick += buttonPcdkGetPose_Click;
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
            string path = Path.Combine(OutputDir, "PcdkPose.csv");
            List<PoseData> poseData = poseBuffer.Memory.ToList();
            Csv.WriteCsv(path, poseData);
        }

        private void buttonPcdkGetJoint_Click(object sender, EventArgs e)
        {
            double?[] joint = getPcdkJoint();
            Time = StopWatch.ElapsedMilliseconds / 1000;
            addToPcdkBuffers(null, joint, Time);
            updatePCDKTxtBox(null, joint);
        }

        private void buttonPcdkAttachJoint_Click(object sender, EventArgs e)
        {
            if (buttonPcdkAttachJoint.Text == "Attach")
            {
                buttonPcdkAttachJoint.Enabled = false;
                FormTimer.Tick += buttonPcdkGetJoint_Click;
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
            List<JointData> jointData = jointBuffer.Memory.ToList();
            Csv.WriteCsv(path, jointData);
        }

        private double?[] getPcdkPose()
        {
            if (radioButtonPcdkWorldFrame.Checked)
            {
                return PCDK.GetPoseWF().Select(x => (double?)x).ToArray();
            }
            else if (radioButtonPcdkUserFrame.Checked)
            {
                return PCDK.GetPoseUF().Select(x => (double?)x).ToArray();
            }
            return null;
        }

        private double?[] getPcdkJoint()
        {

            return PCDK.GetJointPosition().Select(x => (double?)x).ToArray();
        }

        private void addToPcdkBuffers(double?[] pose, double?[] joint, double time)
        { 
            if (pose != null)
            {
                double[] array = pose.Select(x => (double)x).ToArray();
                poseBuffer.Add(new PoseData(array, time));
            }
            if (joint != null)
            {
                double[] array = joint.Select(x => (double)x).ToArray();
                jointBuffer.Add(new JointData(array, time));
            }
        }

        

        // --------------------------------- VXelements

        private void buttonVxQuickConnect_Click(object sender, EventArgs e)
        {
            string targetsPath = Path.Combine(ReferenceDir, "targets.txt");
            string modelPath = Path.Combine(ReferenceDir, "model.txt");
            Vx.QuickConnect(targetsPath, modelPath);
        }

        private void buttonVxStartTracking_Click(object sender, EventArgs e)
        {
            Vx.StartTracking();
        }

        private void buttonVxStopTracking_Click(object sender, EventArgs e)
        {
            Vx.StopTracking();
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
            string[] path = new string[3];
            path[0] = Path.Combine(OutputDir, "VxRaw.csv");
            path[1] = Path.Combine(OutputDir, "VxKf.csv");
            path[2] = Path.Combine(OutputDir, "VxRkf.csv");
            Vx.ExportBuffers(path);
        }

        // --------------------------------- Gui Update

        private void updateVxTxtBox(Vector<double> pose)
        {
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updateVxTxtBox(pose);
                this.Invoke(safeUpdate);
            }
            else
            {
                textBoxVxX.Text = Vx.PoseCameraFrame[0].ToString("0.0000");
                textBoxVxY.Text = Vx.PoseCameraFrame[1].ToString("0.0000");
                textBoxVxZ.Text = Vx.PoseCameraFrame[2].ToString("0.0000");
                textBoxVxAlpha.Text = Vx.PoseCameraFrame[3].ToString("0.0000");
                textBoxVxBeta.Text = Vx.PoseCameraFrame[4].ToString("0.0000");
                textBoxVxGamma.Text = Vx.PoseCameraFrame[5].ToString("0.0000");
            }
        }

        private void updatePCDKTxtBox(double?[] pose, double?[] joint)
        {
            if (this.InvokeRequired)
            {
                Action safeUpdate = () => updatePCDKTxtBox(pose, joint);
                this.Invoke(safeUpdate);
            }
            else
            {
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

        // --------------------------------- Logger

        // --------------------------------- Test
        private void buttonRunMain_Click(object sender, EventArgs e)
        {

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
                textBoxDebug.AppendText(str + Environment.NewLine);
            }
        }

        private void printToDebug(string str)
        {
            textBoxDebug.AppendText(str + Environment.NewLine);
        }




        private void listViewLogger_SelectedIndexChanged(object sender, EventArgs e)
        {

        }


    }



}
