﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;
using MathNet.Numerics.LinearAlgebra;
using FanucPCDK;
using Serilog;


namespace FanucController
{
    /// <summary>
    /// This class contains members regarding the line following experiment.
    /// </summary>
    public class PathTracking
    {
        #region Fields

        // Directories and paths
        public string TopDir;
        public string OutputDir;
        public string ReferenceDir;
        public string ScriptsDir;

        // Timers
        public System.Timers.Timer Timer;
        public Stopwatch Stopwatch;
        public VXelementsUtility Vx;

        // Data
        public Buffer<PoseData> ErrorRawBuffer;
        public Buffer<PoseData> ErrorKfBuffer;
        public Buffer<PoseData> ErrorBuffer;
        public Buffer<PoseData> ControlBuffer;
        public Buffer<PoseData> PoseBuffer;
        public Buffer<PoseData> PoseBufferRaw;
        public Buffer<PoseData> PoseBufferKf;
        public Buffer<PoseData> PoseBufferRkf;
        public Buffer<PoseData> PidControlBuffer;
        public Buffer<PoseData> IlcControlBuffer;
        public Buffer<PoseData> PnnControlBuffer;
        public Buffer<PoseData> PnnErrorBuffer;
        public Buffer<PoseData> MbpoControlBuffer;
        public Buffer<PoseData> BpnnpidControlBuffer;
        public Buffer<PoseData> BpnnpidControlBuffer6D;
        public Buffer<PidCoefficients> BpnnpidCoefficientsBuffer;
        public Buffer<PidCoefficients> BpnnpidCoefficientsBuffer6D;

        // Calibration
        public RotationIdentification RotationId;
        public Dictionary<string, Vector<double>> PoseDict;

        // PCDK
        public string ProgramName;
        public DpmWatcher PositionWatcher;
        public DpmWatcher OrientationWatcher;
        protected bool digOut2;
        protected bool digOut3;
        protected bool progStarted;
        protected bool progStopped;
        protected bool iterStarted;
        protected bool iterStopped;

        // Linear Track
        public LinearPath LinearPath;
        protected readonly object errorLock = new object();
        protected Vector<double> pathErrror;
        public Vector<double> PathError
        {
            get
            {
                lock (errorLock) { return pathErrror; }
            }
            set
            {
                lock (errorLock) { pathErrror = value; }
            }
        }

        // Position or Orientation
        protected bool flagPosition;
        protected bool flagOrientation;

        // Line or Circle
        protected bool flagLineCircle;

        // Iteration
        protected int iterNum;
        protected int iterIndex;
        protected bool stepMode;
        protected List<double> startTimes;
        protected List<double> endTimes;

        // Control
        protected bool flagDpm;
        protected bool flagPControl;
        protected bool flagIlc;
        protected bool flagPNN;
        protected bool flagMBPO;
        protected bool flagBPNNPID;
        public double Time;

        // P Control
        protected Vector<double> uP;
        public Matrix<double> Kp;

        // PID
        protected Vector<double> uPID;
        public PathTrackingPID PidControlPosition;
        public PathTrackingPID PidControlOrientation;

        // ILC
        protected Vector<double> uIlc;
        Matrix<double> KpIlc;
        Matrix<double> KdIlc;
        public List<IterativeLearningControl> IlcList;
        public IterativeLearningControl Ilc;
        public IterativeLearningControl LastIlc;

        // P Neural Network (PNN)
        public PathTrackingPNN Pnn;
        protected Vector<double> uPnn;

        // MBPO
        public PathTrackingMBPO Mbpo;
        public PathTrackingMBPO6D Mbpo6D;
        public PathTrackingMBPO MbpoXYZ;
        public PathTrackingMBPO MbpoWPR;
        protected Vector<double> uMbpo;

        //BPNNPID
        public PathTrackingBPNNPID Bpnnpid;
        public PathTrackingBPNNPID6D Bpnnpid6D;
        protected Vector<double> uBpnnpid;
        protected Vector<double> uBpnnpid6D;

        // Python
        public string ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";

        #endregion

        #region Constructor

        public PathTracking(VXelementsUtility vx, System.Timers.Timer timer, Stopwatch stopwatch, string topDir)
        {
            Vx = vx;
            Timer = timer;
            Stopwatch = stopwatch;
            TopDir = topDir;
            OutputDir = Path.Combine(topDir, "output");
            ReferenceDir = Path.Combine(topDir, "reference");
            ScriptsDir = Path.Combine(topDir, "scripts");
            RotationId = new RotationIdentification();
            RotationId.fileName = "RotationID.json";
            LinearPath = new LinearPath();
            LinearPath.fileName = "LinearPath.json";
            PoseDict = new Dictionary<string, Vector<double>>();

            // Buffers
            ErrorRawBuffer = new Buffer<PoseData>(20_000);
            ErrorKfBuffer = new Buffer<PoseData>(20_000);
            ErrorBuffer = new Buffer<PoseData>(20_000);
            ControlBuffer = new Buffer<PoseData>(20_000);
            PoseBuffer = new Buffer<PoseData>(20_000);
            PoseBufferRaw = new Buffer<PoseData>(20_000);
            PoseBufferKf = new Buffer<PoseData>(20_000);
            PoseBufferRkf = new Buffer<PoseData>(20_000);
            PidControlBuffer = new Buffer<PoseData>(20_000);
            IlcControlBuffer = new Buffer<PoseData>(20_000);
            PnnControlBuffer = new Buffer<PoseData>(20_000);
            PnnErrorBuffer = new Buffer<PoseData>(20_000);
            MbpoControlBuffer = new Buffer<PoseData>(20_000);
            BpnnpidControlBuffer = new Buffer<PoseData>(20_000);
            BpnnpidControlBuffer6D = new Buffer<PoseData>(20_000);
            BpnnpidCoefficientsBuffer = new Buffer<PidCoefficients>(20_000);
            BpnnpidCoefficientsBuffer6D = new Buffer<PidCoefficients>(20_000);

            // Path Error
            startTimes = new List<double>();
            endTimes = new List<double>();
            PathError = CreateVector.Dense<double>(6);

            // Flags
            flagDpm = false;
            flagPControl = false;
            flagIlc = false;
            flagPNN = false;
            flagMBPO = false;

            // P Control
            Kp = CreateMatrix.Dense<double>(6, 6);
            Kp[0, 0] = 0.3; Kp[1, 1] = 0.3; Kp[2, 2] = 0.3;

            // PID Control
            PidControlPosition = new PathTrackingPID()
            {
                Kp = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.4, 0.3, 0.4 }),
                Ki = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.005, 0.005, 0.05 }),
                Kd = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.0005, 0.0005, 0.0030})
            };

            PidControlOrientation = new PathTrackingPID()
            {
                Kp = CreateMatrix.DenseOfDiagonalArray(new double[3] { 180 / Math.PI * 0.3, 180 / Math.PI * 0.15, 180 / Math.PI * 0.45 }),
                Ki = CreateMatrix.DenseOfDiagonalArray(new double[3] { 180 / Math.PI * 0.020, 180 / Math.PI * 0.020, 180 / Math.PI * 0.020 }),
                Kd = CreateMatrix.DenseOfDiagonalArray(new double[3] { 180 / Math.PI * 0.0005, 180 / Math.PI * 0.0005, 180 / Math.PI * 0.0005 })
            };

            // ILC
            IlcList = new List<IterativeLearningControl>();
            KpIlc = CreateMatrix.Dense<double>(6, 6);
            KpIlc[0, 0] = 0.03; KpIlc[1, 1] = 0.03; KpIlc[2, 2] = 0.03;
            KdIlc = CreateMatrix.Dense<double>(6, 6);
            KdIlc[0, 0] = 0; KdIlc[1, 1] = 0; KdIlc[2, 2] = 0;
            Ilc = new IterativeLearningControl(KpIlc, KdIlc);
            LastIlc = new IterativeLearningControl(KpIlc, KdIlc);

            // P Neural Network
            Pnn = new PathTrackingPNN()
            {
                OutputDir = OutputDir
            };

            // MBPO
            Mbpo = new PathTrackingMBPO()
            {
                OutputDir = OutputDir
            };
            Mbpo6D = new PathTrackingMBPO6D()
            {
                OutputDir = OutputDir
            };
            MbpoXYZ = new PathTrackingMBPO()
            {
                OutputDir = OutputDir,
                Actor = "actor_xyz.onnx",
                iterScript = "iter_mbpo_xyz.py",
                plotScript = "iter_mbpo_plot_6d.py",
                ExplorationNoise = new double[6] { 0.005, 0.005, 0.005, 0, 0, 0 },
                OUExplorationNoise = new OrnsteinUlhenbeckNoise(new double[6] { 0.005, 0.005, 0.005, 0, 0, 0 }, new double[] { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 }),
                WarmupNoise = 0.10, // ratio of min/max control
                OUWarmupNoise = new OrnsteinUlhenbeckNoise(new double[] { 0.05, 0.05, 0.05, 0, 0, 0 },
                new double[] { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 }),
                minControl = new double[3] { -0.08, -0.08, -0.08 },
                maxControl = new double[3] { 0.08, 0.08, 0.08 },
                minState = new double[6] { -600, 700, 500, -0.50, -0.50, -0.50 },
                maxState = new double[6] { -300, 1500, 800, 0.50, 0.50, 0.50 },
            };
            MbpoWPR = new PathTrackingMBPO()
            {
                OutputDir = OutputDir,
                Actor = "actor_wpr.onnx",
                iterScript = "iter_mbpo_wpr.py",
                plotScript = "iter_mbpo_plot_6d.py",
                ExplorationNoise = new double[6] { 0, 0, 0, 0.0005, 0.0005, 0.0005 },
                OUExplorationNoise = new OrnsteinUlhenbeckNoise(new double[6] { 0, 0, 0, 0.0005, 0.0005, 0.0005 }, new double[6] { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 }),
                WarmupNoise = 0.10, // ratio of min/max control
                OUWarmupNoise = new OrnsteinUlhenbeckNoise(new double[6] { 0, 0, 0, 0.002, 0.002, 0.002 },
                new double[6] { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 }),
                minControl = new double[3] { -0.003, -0.003, -0.003 },
                maxControl = new double[3] { 0.003, 0.003, 0.003 },
                minState = new double[6] { 2.90, -0.15, 1.4, -0.002, -0.002, -0.002 },
                maxState = new double[6] { 3.20, 0.30, 1.60, 0.002, 0.002, 0.002 },
            };

            // BPNNPID
            Bpnnpid = new PathTrackingBPNNPID();
            Bpnnpid6D = new PathTrackingBPNNPID6D();

        }
        #endregion

        #region Path Tracking Actions

        public virtual void Init(string pathPath, string progName, int iter = 1, bool step = false, bool dpm = false,
            bool pControl = false, bool ilc = false, bool pNN = false, bool mbpo = false, bool bpnnpid = false, 
            bool circle = false, bool position = false, bool orientation = false)
        {
            // Program & Path Info
            LinearPath = LinearPath.FromJson(pathPath);
            RotationId.Rotation = LinearPath.Rotation;
            ProgramName = progName;

            // Flags
            progStarted = false;
            progStopped = false;
            iterStarted = false;
            iterStopped = false;
            digOut2 = false;
            digOut3 = false;

            // Iterations
            iterNum = iter;
            iterIndex = 0;

            // PCDK
            PCDK.SetupDPM(sch: 1);
            PCDK.SetDigitalOutput(2, false);

            // DPM Watcher
            PositionWatcher = new DpmWatcher
            {
                OffsetLimit = 0.5,
                CumulativeOffset = CreateVector.DenseOfArray<double>(new double[3] { 0.0, 0.0, 0.0 }),
                CumulativeOffsetLimit = 5,
                ErrorLimit = 5,
                CumulativeError = CreateVector.Dense<double>(3),
                CumulativeErrorLimit = 5
            };
            OrientationWatcher = new DpmWatcher
            {
                OffsetLimit = 0.40,
                CumulativeOffset = CreateVector.DenseOfArray<double>(new double[3] { 0.0, 0.0, 0.0 }),
                CumulativeOffsetLimit = 10.0,
                ErrorLimit = 0.010,
                CumulativeError = CreateVector.Dense<double>(3),
                CumulativeErrorLimit = 2.00
            };

            // Reset Buffers
            ErrorRawBuffer.Reset();
            ErrorKfBuffer.Reset();
            ErrorBuffer.Reset();
            ControlBuffer.Reset();
            PoseBuffer.Reset();
            PoseBufferRaw.Reset();
            PoseBufferKf.Reset();
            PoseBufferRkf.Reset();
            PidControlBuffer.Reset();
            IlcControlBuffer.Reset();
            PnnControlBuffer.Reset();
            PnnErrorBuffer.Reset();
            MbpoControlBuffer.Reset();
            BpnnpidControlBuffer.Reset();
            BpnnpidControlBuffer6D.Reset();
            BpnnpidCoefficientsBuffer.Reset();
            BpnnpidCoefficientsBuffer6D.Reset();

            // Line or Circle
            flagLineCircle = circle;

            // Position & Orientation
            flagPosition = position;
            flagOrientation = orientation;

            // Step Mode
            stepMode = step;

            // DPM
            flagDpm = dpm;

            // P Control
            flagPControl = pControl;

            // PID
            if (flagPControl)
            {
                PidControlPosition.Init();
                PidControlOrientation.Init();
            }

            // ILC
            flagIlc = ilc;
            if (flagIlc)
            {
                IlcList.Clear();
                Ilc = new IterativeLearningControl(KpIlc, KdIlc);
                LastIlc = new IterativeLearningControl(KpIlc, KdIlc);
            }

            // P Neural Network
            flagPNN = pNN;
            Pnn.Init();

            // MBPO
            flagMBPO = mbpo;
            Mbpo.Init();
            Mbpo6D.Init();
            MbpoXYZ.Init();
            MbpoWPR.Init();

            // BPNNPID
            flagBPNNPID = bpnnpid;
            Bpnnpid.Init();
            Bpnnpid6D.Init();
        }

        public virtual void Reset()
        {
            // Reset flags
            progStarted = false;
            progStopped = false;
            iterStarted = false;
            iterStopped = false;
            digOut2 = false;
            digOut3 = false;

            // Iterations
            //iterIndex = 0;

            // PCDK
            PCDK.SetDigitalOutput(2, false);

            // DPM watcher
            PositionWatcher.Reset();
            OrientationWatcher.Reset();

            // Reset Buffers
            ErrorRawBuffer.Reset();
            ErrorKfBuffer.Reset();
            ErrorBuffer.Reset();
            ControlBuffer.Reset();
            PoseBuffer.Reset();
            PoseBufferRaw.Reset();
            PoseBufferKf.Reset();
            PoseBufferRkf.Reset();
            IlcControlBuffer.Reset();
            PnnControlBuffer.Reset();
            PnnErrorBuffer.Reset();
            MbpoControlBuffer.Reset();
            BpnnpidControlBuffer.Reset();
            BpnnpidCoefficientsBuffer.Reset();
            BpnnpidControlBuffer6D.Reset();
            BpnnpidCoefficientsBuffer6D.Reset();

            // Reset times
            //startTimes.Clear();
            //endTimes.Clear();

            // PID
            if (flagPControl)
            {
                PidControlPosition.Reset();
                PidControlOrientation.Reset();
            }

            // ILC
            if (flagIlc)
            {
                LastIlc = Ilc;
                Ilc = new IterativeLearningControl(KpIlc, KdIlc);
                if (iterIndex == 0)
                {
                    Ilc.Init();
                }
                else
                {
                    Ilc.Init(LastIlc);
                }
            }

            // P Neural Network
            if (flagPNN)
            {
                Pnn.Reset();
            }

            // MBPO
            if (flagMBPO)
            {
                Mbpo.Reset();
                Mbpo6D.Reset();
                MbpoXYZ.Reset();
                MbpoWPR.Reset();
            }

            // BPNNPID
            if (flagBPNNPID)
            {
                Bpnnpid.Reset();
                Bpnnpid6D.Reset();
            }
        }

        public virtual void Start()
        {
            // Reset
            Reset();

            // Base
            if (!Timer.Enabled)
            {
                Timer.Start();
            }
            PCDK.Run(this.ProgramName);
            Timer.Elapsed += Loop;
            //Log.Information("Line tracking started.");
        }

        public virtual void Loop(object s, EventArgs ev)
        {
            // Monitor program
            progStarted = PCDK.IsRunning();
            progStopped = PCDK.IsAborted();
            digOut3 = PCDK.GetDigitalOutput(3);

            // Monital Digital IO
            if (digOut3 && progStarted && !iterStarted)
            {
                digOut2 = true;
                PCDK.SetDigitalOutput(2, digOut2);
                iterStarted = true;
                startTimes.Add(Stopwatch.Elapsed.TotalSeconds);
            }
            if (!iterStopped && progStopped)
            {
                digOut2 = false;
                PCDK.SetDigitalOutput(2, digOut2);
                iterStopped = true;
                endTimes.Add(Stopwatch.Elapsed.TotalSeconds);
            }

            // One time step action
            if (iterStarted && !iterStopped)
            {
                // Get data and compute path error
                Time = Stopwatch.Elapsed.TotalSeconds - startTimes[iterIndex];
                Vector<double> xRaw = GetVxUFPose(mode: "raw");
                Vector<double> xKf = GetVxUFPose(mode: "kf");
                Vector<double> xRkf = GetVxUFPose(mode: "rkf");
                Vector<double> x = xRkf;
                //Vector<double> closestP = MathLib.PointToLinePoint(LinearPath.PointStart.SubVector(0,3), LinearPath.PointEnd.SubVector(0,3), x.SubVector(0, 3));
                //Vector<double> xd = (closestP.ToColumnMatrix().Stack(x.SubVector(3, 3).ToColumnMatrix())).Column(0);
                Vector<double> xd;
                if (!flagLineCircle) // Line or Circle Tracking
                {
                    xd = CalculateDesiredPose(x); // Line Tracking
                }
                else
                {
                    xd = x;
                }
                
                PathError = xd - x;
                Vector<double> u = CreateVector.Dense<double>(6);
                Vector<double> uPosition = CreateVector.Dense<double>(6);
                Vector<double> uOrientation = CreateVector.Dense<double>(6);


                // P Control
                if (flagPControl)
                {
                    //uP = Pid(PathError);
                    uPID = PidControlPosition.Control(PathError.SubVector(0, 3));
                    uPID += PidControlOrientation.Control(PathError.SubVector(3, 3), wpr: true);
                    u += uPID;
                }

                // ILC
                if (flagIlc)
                {
                    Ilc.Add(PathError, Time);
                    uIlc = Ilc.Query(Time);
                    u += uIlc;
                }

                // PNN
                if (flagPNN)
                {

                    uPnn = Pnn.Control(x, u, Time, rand: true);
                    u += uPnn;
                }

                // MBPO
                if (flagMBPO)
                {
                    //uMbpo = Mbpo.Control(x, PathError, rand: true);
                    //uMbpo = Mbpo6D.Control(x, PathError, rand: true);
                    uMbpo = MbpoXYZ.ControlXYZ(x, PathError, rand: true) + MbpoWPR.ControlWPR(x, PathError, rand:true);
                    u += uMbpo;
                }

                // BPNNPID
                if (flagBPNNPID)
                {
                    uBpnnpid = Bpnnpid.Control(xd.SubVector(0, 3), x.SubVector(0, 3), PathError.SubVector(0, 3));
                    uBpnnpid6D = Bpnnpid6D.Control(xd, x, PathError);
                    u += uBpnnpid;
                }

                // Position & Orientation
                if (flagPosition && !flagOrientation)
                {
                    u.ClearSubVector(3, 3);
                }
                else if (!flagPosition && flagOrientation)
                {
                    u.ClearSubVector(0, 3);
                }

                // Dpm fuckery
                var u_temp = u.Clone();
                u[3] = -u_temp[4];
                u[4] = u_temp[3];

                // Dpm
                if (PositionWatcher.LimitCheck(PathError.SubVector(0, 3), u.SubVector(0, 3)))
                {
                    if (OrientationWatcher.LimitCheck(PathError.SubVector(3, 3), u.SubVector(3, 3)))
                    {
                        if (flagDpm)
                        {
                            PCDK.ApplyDPM(u.AsArray(), 1);
                        }
                    }
                }

                u = u_temp;

                // Save data to buffer
                ErrorRawBuffer.Add(new PoseData((xd - xRaw).AsArray(), Time));
                ErrorKfBuffer.Add(new PoseData((xd - xKf).AsArray(), Time));
                ErrorBuffer.Add(new PoseData(PathError.AsArray(), Time));
                ControlBuffer.Add(new PoseData(u.AsArray(), Time));
                PoseBuffer.Add(new PoseData(x.AsArray(), Time));
                PoseBufferRaw.Add(new PoseData(xRaw.AsArray(), Time));
                PoseBufferKf.Add(new PoseData(xKf.AsArray(), Time));
                PoseBufferRkf.Add(new PoseData(xRkf.AsArray(), Time));
                if (flagPControl)
                {
                    PidControlBuffer.Add(new PoseData(uPID.AsArray(), Time));
                }
                if (flagIlc)
                {
                    IlcControlBuffer.Add(new PoseData(uIlc.AsArray(), Time));
                }
                if (flagPNN)
                {
                    PnnControlBuffer.Add(new PoseData(uPnn.AsArray(), Time));
                    PnnErrorBuffer.Add(new PoseData(Pnn.PnnError.AsArray(), Time));
                }
                if (flagMBPO)
                {
                    MbpoControlBuffer.Add(new PoseData(uMbpo.AsArray(), Time));
                }
                if (flagBPNNPID)
                {
                    BpnnpidControlBuffer.Add(new PoseData(uBpnnpid.AsArray(), Time));
                    BpnnpidCoefficientsBuffer.Add(new PidCoefficients(Bpnnpid.kValues, Time));
                    BpnnpidControlBuffer6D.Add(new PoseData(uBpnnpid6D.AsArray(), Time));
                    BpnnpidCoefficientsBuffer6D.Add(new PidCoefficients(Bpnnpid6D.kValues, Time));
                }

            }
            // Stop program if condition is reached
            if (iterStopped)
            {
                Stop();
            }

        }

        public virtual void Stop()
        {
            // Timer
            Timer.Elapsed -= Loop;

            // Create Iter dir
            string iterDir = Path.Combine(OutputDir, $"iteration_{iterIndex}");
            Directory.CreateDirectory(iterDir);

            // Export VXelements Data
            var VxPaths = new string[3];
            VxPaths[0] = Path.Combine(iterDir, "VxRaw.csv");
            VxPaths[1] = Path.Combine(iterDir, "VxKf.csv");
            VxPaths[2] = Path.Combine(iterDir, "VxRkf.csv");
            Vx.ExportBuffers(VxPaths);

            // Export buffer data
            string path;
            path = Path.Combine(iterDir, "LineTrackRawError.csv");
            Csv.WriteCsv<PoseData>(path, ErrorRawBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackKfError.csv");
            Csv.WriteCsv<PoseData>(path, ErrorKfBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackError.csv");
            Csv.WriteCsv<PoseData>(path, ErrorBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackControl.csv");
            Csv.WriteCsv(path, ControlBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackPose.csv");
            Csv.WriteCsv(path, PoseBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackPoseRaw.csv");
            Csv.WriteCsv(path, PoseBufferRaw.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackPoseKf.csv");
            Csv.WriteCsv(path, PoseBufferKf.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackPoseRkf.csv");
            Csv.WriteCsv(path, PoseBufferRkf.Memory.ToList());
            Thread.Sleep(1000);

            // Plot iteration data in python
            string savePath = Path.Combine(iterDir, $"iteration_fig_{iterIndex}");
            string[] args = new string[2] { iterDir, savePath };
            args = new string[2] { iterDir, savePath };
            //string scriptPath = Path.Combine(ScriptDir, "ilc_plot.py");
            PythonScripts.RunParallel("iter_plot.py", args: args);

            // Pid
            if (flagPControl)
            {
                path = Path.Combine(iterDir, "LineTrackPidControl.csv");
                Csv.WriteCsv(path, PidControlBuffer.Memory.ToList());
            }

            // ILC
            if (flagIlc)
            {
                Ilc.PControl();
                //path = Path.Combine(iterDir, "LineTrackIlcControl.csv");
                //Ilc.ToCsv(path);
                path = Path.Combine(iterDir, "LineTrackIlcControl.csv");
                Csv.WriteCsv(path, IlcControlBuffer.Memory.ToList());
                Thread.Sleep(200);

                // Plot iteration data in python
                savePath = Path.Combine(iterDir, $"iteration_ilc_fig_{iterIndex}");
                Ilc.Iteration(savePath, iterDir);
            }

            // PNN
            if (flagPNN)
            {
                // Write PNN control CSV file
                path = Path.Combine(iterDir, "LineTrackPnnControl.csv");
                Csv.WriteCsv(path, PnnControlBuffer.Memory.ToList());

                // Write PNN error CSV file
                path = Path.Combine(iterDir, "LineTrackPnnError.csv");
                Csv.WriteCsv(path, PnnErrorBuffer.Memory.ToList());

                // PNN Plot
                Pnn.Plot(iterDir: iterDir);

                // PNN Iteration
                List<string> dataDirs = new List<string>()
                {
                    OutputDir
                };
                Pnn.Iteration(nEpoch: Pnn.NEpochs, dataDirs: dataDirs);

            }

            // MBPO
            if (flagMBPO)
            {
                // Write MBPO control CSV file
                path = Path.Combine(iterDir, "LineTrackMbpoControl.csv");
                Csv.WriteCsv(path, MbpoControlBuffer.Memory.ToList());

                // PNN Plot
                //Mbpo.Plot(iterDir: iterDir);
                //Mbpo6D.Plot(iterDir: iterDir);
                MbpoXYZ.Plot(iterDir: iterDir);

                // MBPO Iteration
                List<string> dataDirs = new List<string>()
                {
                    OutputDir
                };
                //Mbpo.Iteration(nEpoch: Mbpo.NEpochs, gradSteps: Mbpo.GradientSteps, iterDir: iterDir, dataDirs: dataDirs);
                //Mbpo6D.Iteration(nEpoch: Mbpo6D.NEpochs, gradSteps: Mbpo6D.GradientSteps, iterDir: iterDir, dataDirs: dataDirs);
                MbpoXYZ.Iteration(nEpoch: MbpoXYZ.NEpochs, gradSteps: MbpoXYZ.GradientSteps, iterDir: iterDir, dataDirs: dataDirs);
                MbpoWPR.Iteration(nEpoch: MbpoWPR.NEpochs, gradSteps: MbpoWPR.GradientSteps, iterDir: iterDir, dataDirs: dataDirs, script:false);
                //Thread.Sleep(50);
                //MbpoWPR.Iteration(nEpoch: MbpoWPR.NEpochs, gradSteps: MbpoWPR.GradientSteps, iterDir: iterDir, dataDirs: dataDirs);
            }

            // BPNNPID
            if (flagBPNNPID)
            {
                // Write PBNNPID control CSV file
                path = Path.Combine(iterDir, "LineTrackBpnnpidControl.csv");
                Csv.WriteCsv(path, BpnnpidControlBuffer.Memory.ToList());
                path = Path.Combine(iterDir, "LineTrackBpnnpidCoefficients.csv");
                Csv.WriteCsv(path, BpnnpidCoefficientsBuffer.Memory.ToList());
                path = Path.Combine(iterDir, "LineTrackBpnnpidControl6D.csv");
                Csv.WriteCsv(path, BpnnpidControlBuffer6D.Memory.ToList());
                path = Path.Combine(iterDir, "LineTrackBpnnpidCoefficients6D.csv");
                Csv.WriteCsv(path, BpnnpidCoefficientsBuffer6D.Memory.ToList());
            }

            // Log
            // Log.Information($"Iteration {iterIndex} completed.");

            // Restart 
            iterIndex++;
            if (iterIndex < iterNum)
            {
                if (stepMode)
                {
                    MessageBox.Show("Continue?");
                }
                Start();
            }
        }

        public bool Check()
        {
            return true;
        }

        #endregion

        #region Control Algorithms

        public Vector<double> Pid(Vector<double> error)
        {
            return Kp * error;
        }

        public Vector<double> CalculateDesiredPose(Vector<double> pose)
        {
            // Position

            Vector<double> A = LinearPath.PointStart.SubVector(0, 3);
            Vector<double> B = LinearPath.PointEnd.SubVector(0, 3);
            Vector<double> P = pose.SubVector(0, 3);
            double L = (B - A).L2Norm(); // path length
            Vector<double> normal = (B - A) / L;
            Vector<double> Pd = A + ((P - A).DotProduct(normal)) * normal;
            double Lp = (Pd - A).L2Norm(); // path length progressed

            // Orientation
            Vector<double> WprA = LinearPath.PointStart.SubVector(3, 3);
            Vector<double> WprB = LinearPath.PointEnd.SubVector(3, 3);
            //Vector<double> WprP = pose.SubVector(3, 3);
            Vector<double> WprPd = WprA + (WprB - WprA) * Lp / L;

            Vector<double> desiredPose = (Pd.ToColumnMatrix().Stack(WprPd.ToColumnMatrix())).Column(0);

            return desiredPose;

        }

        #endregion

        #region Data Acquisition & Processing

        public Vector<double> GetFanucPose()
        {
            return CreateVector.DenseOfArray(PCDK.GetPoseUF());
        }

        public void CalculateRotaionWPR(int sampleNum)
        {
            Vector<double> cameraPose;
            Vector<double> robotPose;
            Vector<double> cameraPoseSum = CreateVector.Dense<double>(6);
            Vector<double> robotPoseSum = CreateVector.Dense<double>(6);
            for (int i = 0; i < sampleNum; i++)
            {
                cameraPose = Vx.PoseCameraFrameRkf;
                cameraPoseSum = CreateVector.DenseOfEnumerable(cameraPoseSum.Zip(cameraPose, (x, y) => x + y));
                robotPose = GetFanucPose();
                robotPoseSum = CreateVector.DenseOfEnumerable(robotPoseSum.Zip(robotPose, (x, y) => x + y));
                System.Threading.Thread.Sleep(50);
            }

            cameraPose = CreateVector.DenseOfEnumerable(cameraPoseSum.Select(x => x / sampleNum));
            robotPose = CreateVector.DenseOfEnumerable(robotPoseSum.Select(x => x * Math.PI / 180 / sampleNum));

            var rotationCamera = MathLib.RotationMatrix(cameraPose[5], cameraPose[4], cameraPose[3]);
            //var rotationRobot = MathLib.RotationMatrix(robotPose[5], robotPose[4], robotPose[3]);
            //var rotationRobot = MathLib.RotationX(robotPose[3]) * MathLib.RotationY(robotPose[4]) * MathLib.RotationZ(robotPose[5]);
            var rotationRobot = MathLib.RotationZ(robotPose[5]) * MathLib.RotationY(robotPose[4]) * MathLib.RotationX(robotPose[3]);
            RotationId.RotationWpr = rotationRobot * rotationCamera.Transpose();
        }

        public void CalcualteRotationOffset(int sampleNum)
        {
            Vector<double> cameraPose;
            Vector<double> robotPose;
            Vector<double> cameraPoseSum = CreateVector.Dense<double>(6);
            Vector<double> robotPoseSum = CreateVector.Dense<double>(6);
            for (int i = 0; i < sampleNum; i++)
            {
                cameraPose = Vx.PoseCameraFrameRkf;
                cameraPoseSum = CreateVector.DenseOfEnumerable(cameraPoseSum.Zip(cameraPose, (x, y) => x + y));
                robotPose = GetFanucPose();
                robotPoseSum = CreateVector.DenseOfEnumerable(robotPoseSum.Zip(robotPose, (x, y) => x + y));
                System.Threading.Thread.Sleep(50);
            }


            cameraPose = CreateVector.DenseOfEnumerable(cameraPoseSum.Select(x => x / sampleNum));
            robotPose = CreateVector.DenseOfEnumerable(robotPoseSum.Select(x => x / sampleNum));

            RotationId.PositionOffset = robotPose.SubVector(0, 3) -  RotationId.Rotation * cameraPose.SubVector(0, 3);

            robotPose[3] = robotPose[3] * Math.PI / 180;
            robotPose[4] = robotPose[4] * Math.PI / 180;
            robotPose[5] = robotPose[5] * Math.PI / 180;

            var rotationCamera = MathLib.RotationZ(cameraPose[5]) * MathLib.RotationY(cameraPose[4]) * MathLib.RotationX(cameraPose[3]);
            var rotationRobot = MathLib.RotationZ(robotPose[5]) * MathLib.RotationY(robotPose[4]) * MathLib.RotationX(robotPose[3]);
            //var rotationCamera = MathLib.RotationMatrix(cameraPose[5], cameraPose[4], cameraPose[3]);
            //var rotationRobot = MathLib.RotationMatrix(robotPose[5], robotPose[4], robotPose[3]);


            //RotationId.RotationOffset = rotationRobot * (RotationId.Rotation * rotationCamera).Transpose();
            //RotationId.RotationOffset = RotationId.Rotation.Transpose() * rotationRobot * rotationCamera.Transpose();
            //RotationId.RotationOffset = rotationRobot * rotationCamera.Transpose() * RotationId.Rotation.Transpose();
            RotationId.RotationOffset = rotationCamera.Transpose() * RotationId.Rotation.Transpose() * rotationRobot;
        }

        public Vector<double> GetVxCameraPose(string mode = "rkf")
        {
            switch (mode)
            {
                case "rkf":
                    // RKF Pose
                    return Vx.PoseCameraFrameRkf;
                case "kf":
                    // KF Pose
                    return Vx.PoseCameraFrameKf;
                case "raw":
                    // KF Pose
                    return Vx.PoseCameraFrame;
            }
            // RKF Pose
            return Vx.PoseCameraFrameRkf;
        }

        public Vector<double> GetVxCameraPose(int sampleNum)
        {
            Vector<double> poseSum = CreateVector.Dense<double>(6);
            Vector<double> pose;
            for (int i = 0; i < sampleNum; i++)
            {
                pose = Vx.PoseCameraFrameRkf;
                poseSum = CreateVector.DenseOfEnumerable(poseSum.Zip(pose, (x, y) => x + y));
                System.Threading.Thread.Sleep(50);
            }
            poseSum = CreateVector.DenseOfEnumerable(poseSum.Select(x => x / sampleNum));
            return poseSum;
        }

        public Vector<double> GetVxUFPose()
        {
            var cameraPose = VxCameraToUF(GetVxCameraPose(), RotationId.Rotation);
            return cameraPose;
        }

        public Vector<double> GetVxUFPose(string mode)
        {
            var cameraPose = VxCameraToUF(GetVxCameraPose(mode), RotationId.Rotation);
            return cameraPose;
        }

        public Vector<double> GetVxUFPose(int sampleNum)
        {
            var cameraPose = VxCameraToUF(GetVxCameraPose(sampleNum), RotationId.Rotation);
            return cameraPose;
        }

        // ------------------------------------------- Helper
        public Vector<double> VxCameraToUF(Vector<double> cameraPose, Matrix<double> rotationCameraUF)
        {
            // Apply transformation to cameraPose.
            var cameraPosition = cameraPose.SubVector(0, 3);
            var rotationCamera = MathLib.RotationMatrix(cameraPose[5], cameraPose[4], cameraPose[3]);
            var rotationUF = RotationId.Rotation * rotationCamera * RotationId.RotationOffset;
            cameraPosition = rotationCameraUF * cameraPosition + RotationId.PositionOffset;
            var orientation = CreateVector.DenseOfArray(MathLib.FixedAnglesIkine(rotationUF));
            //var orientation = CreateVector.DenseOfArray(MathLib.EulerAnglesIkine(rotationUF));
            return (cameraPosition.ToColumnMatrix().Stack(orientation.ToColumnMatrix())).Column(0);
        }

        #endregion

        #region Rotation and Linear Path Identification

        public void RecordPose(string key, Vector<double> pose)
        {
            /* Rotation keys: idX1, idX2, idY1, idY2, idZ1, idZ2, testP1, testP2
             * Linear path keys: startP, endP    
             */
            PoseDict[key] = pose;
        }

        public void RecordIdentificationPose(ref Vector<double> idPose, int sampleNum)
        {
            idPose = GetVxCameraPose(sampleNum);
        }

        public void RotationIdentify(string xyz)
        {
            switch (xyz)
            {
                case "xy":
                    RotationId.x[0] = PoseDict["idX1"]; RotationId.x[1] = PoseDict["idX2"];
                    RotationId.y[0] = PoseDict["idY1"]; RotationId.x[1] = PoseDict["idY2"];
                    RotationId.CalculateRotationMatrixXY();
                    RotationId.ToJson(ReferenceDir);
                    break;
                case "xz":
                    RotationId.x[0] = PoseDict["idX1"]; RotationId.x[1] = PoseDict["idX2"];
                    RotationId.z[0] = PoseDict["idZ1"]; RotationId.z[1] = PoseDict["idZ2"];
                    RotationId.CalculateRotationMatrixXZ();
                    RotationId.ToJson(ReferenceDir);
                    break;
                case "yz":
                    RotationId.y[0] = PoseDict["idY1"]; RotationId.y[1] = PoseDict["idY2"];
                    RotationId.z[0] = PoseDict["idZ1"]; RotationId.z[1] = PoseDict["idZ2"];
                    RotationId.CalculateRotationMatrixYZ();
                    RotationId.ToJson(ReferenceDir);
                    break;
            }

        }

        #endregion
    }

    public class DpmWatcher
    {
        // This class watches the DPM offset values and check if they surpass a certain threshold.

        public double OffsetLimit;
        public Vector<double> CumulativeOffset;
        public double CumulativeOffsetLimit;
        public double ErrorLimit;
        public Vector<double> CumulativeError;
        public double CumulativeErrorLimit;

        public void Reset()
        {
            CumulativeError.Clear();
            CumulativeOffset.Clear();
        }

        public bool LimitCheck(Vector<double> error, Vector<double> offset)
        {
            CumulativeError += error;
            CumulativeOffset += offset;
            if (error.L2Norm() > ErrorLimit || CumulativeError.L2Norm() > CumulativeErrorLimit ||
                offset.L2Norm() > OffsetLimit || CumulativeOffset.L2Norm() > CumulativeOffsetLimit)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }

    public class RotationIdentification
    {
        // This class encapsulates the identification of the rotation matrix from C-Track Sensor frame to FANUC user frame.

        [JsonIgnore] public Vector<double>[] x = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] y = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] z = new Vector<double>[2];
        [JsonIgnore] public Matrix<double> Rotation;
        [JsonIgnore] public Matrix<double> RotationWpr;
        [JsonIgnore] public Matrix<double> RotationOffset;
        [JsonIgnore] public Vector<double> PositionOffset;
        [JsonIgnore] public string fileName;

        public void CalculateRotationMatrixXY()
        {
            var i = (x[1] - x[0]).SubVector(0, 3);
            var j = (y[1] - y[0]).SubVector(0, 3);
            var k = MathLib.CrossProduct(i, j);
            Rotation = MathLib.RotationMatrix(i, j, k).Transpose();
        }

        public void CalculateRotationMatrixXZ()
        {
            var i = (this.x[1] - this.x[0]).SubVector(0, 3);
            var k = (this.z[1] - this.z[0]).SubVector(0, 3);
            var j = MathLib.CrossProduct(k, i);
            Rotation = MathLib.RotationMatrix(i, j, k).Transpose();
        }

        public void CalculateRotationMatrixYZ()
        {
            var j = (this.y[1] - this.y[0]).SubVector(0, 3);
            var k = (this.z[1] - this.z[0]).SubVector(0, 3);
            var i = MathLib.CrossProduct(j, k);
            Rotation = MathLib.RotationMatrix(i, j, k).Transpose();
        }

        public double[] RotationArray
        {
            get { return Rotation.ToColumnMajorArray(); }
            set { Rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
        }

        public double[] RotationWprArray
        {
            get { return RotationWpr.ToColumnMajorArray(); }
            set { RotationWpr = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
        }

        public double[] RotationOffsetArray
        {
            get { return RotationOffset.ToColumnMajorArray(); }
            set { RotationOffset = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
        }

        public double[] PositionOffsetArray
        {
            get { return PositionOffset.ToArray(); }
            set { PositionOffset = CreateVector.DenseOfArray<double>(value); }
        }

        public void ReadJson(string dir)
        {
            string path = Path.Combine(dir, fileName);
            string jsonString = File.ReadAllText(path);
            var x = JsonSerializer.Deserialize<RotationIdentification>(jsonString);
            double[] array = JsonSerializer.Deserialize<RotationIdentification>(jsonString).RotationArray;
            double[] arrayWpr = JsonSerializer.Deserialize<RotationIdentification>(jsonString).RotationWprArray;
            double[] arrayOffset = JsonSerializer.Deserialize<RotationIdentification>(jsonString).RotationOffsetArray;
            double[] posArrayOffset = JsonSerializer.Deserialize<RotationIdentification>(jsonString).PositionOffsetArray;
            Rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, array);
            RotationWpr = CreateMatrix.DenseOfColumnMajor<double>(3, 3, arrayWpr);
            RotationOffset = CreateMatrix.DenseOfColumnMajor<double>(3, 3, arrayOffset);
            PositionOffset = CreateVector.DenseOfArray<double>(posArrayOffset);
        }

        public static RotationIdentification FromJson(string path)
        {
            string jsonString = File.ReadAllText(path);
            return JsonSerializer.Deserialize<RotationIdentification>(jsonString);
        }

        public void ToJson(string dir)
        {
            string path = Path.Combine(dir, fileName);
            var options = new JsonSerializerOptions { WriteIndented = true };
            string stringJson = JsonSerializer.Serialize(this, options);
            File.WriteAllText(path, stringJson);
        }

        public static void WriteJson(string path, Matrix<double> rotation)
        {
            RotationIdentification rotationIdentification = new RotationIdentification();
            rotationIdentification.Rotation = rotation;
            var options = new JsonSerializerOptions { WriteIndented = true };
            string stringJson = JsonSerializer.Serialize(rotationIdentification, options);
            File.WriteAllText(path, stringJson);
        }
    }

    public class LinearPath
    {
        // This class represents a linear path.

        [JsonIgnore] public Matrix<double> Rotation; //Rotation Camera -> UF
        [JsonIgnore] public Vector<double> PointStart;
        [JsonIgnore] public Vector<double> PointEnd;
        [JsonIgnore] public string fileName;

        public double[] RotationArray
        {
            get { return Rotation.ToColumnMajorArray(); }
            set { Rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
        }

        public double[] PointStartArray
        {
            get { return PointStart.ToArray(); }
            set { PointStart = CreateVector.DenseOfArray<double>(value); }
        }

        public double[] PointEndArray
        {
            get { return PointEnd.ToArray(); }
            set { PointEnd = CreateVector.DenseOfArray<double>(value); }
        }

        public void GetData(Dictionary<string, Vector<double>> poseDict, RotationIdentification rotId)
        {
            PointStart = poseDict["startP"];
            PointEnd = poseDict["endP"];
            Rotation = rotId.Rotation;
        }

        public void ReadJson(string outDir)
        {
            string path = Path.Combine(outDir, fileName);
            string jsonString = File.ReadAllText(path);
            LinearPath temp = JsonSerializer.Deserialize<LinearPath>(jsonString);
            RotationArray = temp.RotationArray;
            PointStartArray = temp.PointStartArray;
            PointEndArray = temp.PointEndArray;
        }

        public static LinearPath FromJson(string path)
        {
            string jsonString = File.ReadAllText(path);
            return JsonSerializer.Deserialize<LinearPath>(jsonString);
        }

        public void ToJson(string outDir)
        {
            string path = Path.Combine(outDir, fileName);
            var options = new JsonSerializerOptions { WriteIndented = true };
            string stringJson = JsonSerializer.Serialize(this, options);
            File.WriteAllText(path, stringJson);
        }

        public static void WriteJson(string path, Matrix<double> rotation, Vector<double> pointStart, Vector<double> pointEnd)
        {
            LinearPath arrayJson = new LinearPath()
            {
                Rotation = rotation,
                PointStart = pointStart,
                PointEnd = pointEnd
            };
            var options = new JsonSerializerOptions { WriteIndented = true };
            string stringJson = JsonSerializer.Serialize(arrayJson, options);
            File.WriteAllText(path, stringJson);
        }

    }

}
