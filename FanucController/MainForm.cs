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
        public const string TopDir = @"D:\.NET Test";
        public string LoggerTestPath = Path.Combine(TopDir, "LoggerTest", "log_test_1.txt");
        public System.Timers.Timer Timer;
        public Stopwatch StopWatch;
        internal VXelementsUtility Vx;
        public MemorySink Memory;

        // --------------------------------- Constructor
        public MainForm()
        {
            InitializeComponent();

            //Timer
            this.Timer = new System.Timers.Timer();
            this.Timer.Interval = 100;
            this.Timer.AutoReset = true;
            this.Timer.Elapsed += async (s, e) => await TestEventHandlerAsync();
            //this.Timer.Start();

            // Stop watch
            this.StopWatch = new Stopwatch();

            // Initialize VXelement API
            this.Vx = new VXelementsUtility();

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
            this.Memory = MemorySink.Instance;

            

        }

        public async Task TestEventHandlerAsync()
        {
            await Task.Delay(3000);
            Console.WriteLine("Hello");
        }

        // --------------------------------- Fanuc Experiment

        // --------------------------------- PCDK

        // --------------------------------- VXelemets

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
                set { rotation = CreateMatrix.DenseOfColumnMajor<double>(3,3,value); }
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


        private void buttonVxQuickConnect_Click(object sender, EventArgs e)
        {

        }

        private void listViewLogger_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void buttonResetVXtrack_Click(object sender, EventArgs e)
        {

        }
    }



}
