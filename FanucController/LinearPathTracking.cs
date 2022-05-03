using System;
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
    public class LinearPathTracking
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
        public Buffer<PoseData> ErrorBuffer;
        public Buffer<PoseData> ControlBuffer;
        public Buffer<PoseData> PoseBuffer;
        public Buffer<PoseData> IlcControlBuffer;
        public Buffer<PoseData> PnnControlBuffer;
        public Buffer<PoseData> MbpoControlBuffer;

        // Calibration
        public RotationIdentification RotationId;
        public Dictionary<string, Vector<double>> PoseDict;

        // PCDK
        public string ProgramName;
        public DpmWatcher Watcher;
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
        public double Time;

        // P Control
        protected Vector<double> uP;
        public Matrix<double> Kp;

        // ILC
        protected Vector<double> uIlc;
        Matrix<double> KpIlc;
        Matrix<double> KdIlc;
        public List<IterativeLearningControl> IlcList;
        public IterativeLearningControl Ilc;
        public IterativeLearningControl LastIlc;

        // P Neural Network (PNN)
        public LinearPathTrackingPNN Pnn;
        protected Vector<double> uPnn;

        // MBPO
        public LinearPathTrackingMBPO Mbpo;
        protected Vector<double> uMbpo;

        // Python
        public string ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";

        #endregion

        #region Constructor

        public LinearPathTracking(VXelementsUtility vx, System.Timers.Timer timer, Stopwatch stopwatch, string topDir)
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
            ErrorBuffer = new Buffer<PoseData>(20_000);
            ControlBuffer = new Buffer<PoseData>(20_000);
            PoseBuffer = new Buffer<PoseData>(20_000);
            IlcControlBuffer = new Buffer<PoseData>(20_000);
            PnnControlBuffer = new Buffer<PoseData>(20_000);
            MbpoControlBuffer = new Buffer<PoseData>(20_000);

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
            Kp[0,0] = 0.3; Kp[1, 1] = 0.3; Kp[2, 2] = 0.3;

            // ILC
            IlcList = new List<IterativeLearningControl>();
            KpIlc = CreateMatrix.Dense<double>(6, 6);
            KpIlc[0, 0] = 0.03; KpIlc[1, 1] = 0.03; KpIlc[2, 2] = 0.03;
            KdIlc = CreateMatrix.Dense<double>(6, 6);
            KdIlc[0, 0] = 0; KdIlc[1, 1] = 0; KdIlc[2, 2] = 0;
            Ilc = new IterativeLearningControl(KpIlc, KdIlc);
            LastIlc = new IterativeLearningControl(KpIlc, KdIlc);

            // P Neural Network
            Pnn = new LinearPathTrackingPNN()
            {
                OutputDir = OutputDir
            };

            // MBPO
            Mbpo = new LinearPathTrackingMBPO()
            {
                OutputDir = OutputDir
            };


        }
        #endregion

        #region Path Tracking Actions

        public virtual void Init(string pathPath, string progName, int iter=1, bool step = false, bool dpm=false, 
            bool pControl=false, bool ilc=false, bool pNN=false, bool mbpo=false)
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
            PCDK.SetupDPM(sch:1);
            PCDK.SetDigitalOutput(2, false);
            
            // DPM Watcher
            Watcher = new DpmWatcher
            {
                OffsetLimit = 0.5,
                CumulativeOffset = CreateVector.DenseOfArray<double>(new double[3] { 0.0, 0.0, 0.0 }),
                CumulativeOffsetLimit = 5,
                ErrorLimit = 5,
                CumulativeError = CreateVector.Dense<double>(3),
                CumulativeErrorLimit = 5
            };
            
            // Reset Buffers
            ErrorBuffer.Reset();
            ControlBuffer.Reset();
            PoseBuffer.Reset();
            IlcControlBuffer.Reset();
            PnnControlBuffer.Reset();
            MbpoControlBuffer.Reset();

            // Step Mode
            stepMode = step;

            // DPM
            flagDpm = dpm;

            // P Control
            flagPControl = pControl;

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
            Watcher.Reset();
            
            // Reset Buffers
            ErrorBuffer.Reset();
            ControlBuffer.Reset();
            PoseBuffer.Reset();
            IlcControlBuffer.Reset();
            PnnControlBuffer.Reset();
            MbpoControlBuffer.Reset();

            // Reset times
            //startTimes.Clear();
            //endTimes.Clear();

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
                Vector<double> x = GetVxUFPose();
                Vector<double> closestP = MathLib.PointToLinePoint(LinearPath.PointStart.SubVector(0,3), LinearPath.PointEnd.SubVector(0,3), x.SubVector(0, 3));
                Vector<double> xd = (closestP.ToColumnMatrix().Stack(x.SubVector(3, 3).ToColumnMatrix())).Column(0);
                PathError = xd - x;
                Vector<double> u = CreateVector.Dense<double>(6);

                // P Control
                if (flagPControl)
                {
                    uP = Pid(PathError);
                    u += uP;
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
                    
                    uPnn = Pnn.Control(u, Time, rand:false);
                    u += uPnn;
                }

                // MBPO
                if (flagMBPO)
                {
                    uMbpo = Mbpo.Control(PathError, Time, rand:true);
                    // u += uMbpo;
                }


                // Dpm
                if (Watcher.LimitCheck(PathError.SubVector(0,3), u.SubVector(0,3)))
                {
                    if (flagDpm)
                    {
                        PCDK.ApplyDPM(u.AsArray(), 1);
                    }
                }
                // Save data to buffer
                ErrorBuffer.Add(new PoseData(PathError.AsArray(), Time));
                ControlBuffer.Add(new PoseData(u.AsArray(), Time));
                PoseBuffer.Add(new PoseData(x.AsArray(), Time));
                if (flagIlc)
                {
                    IlcControlBuffer.Add(new PoseData(uIlc.AsArray(), Time));
                }
                if (flagPNN)
                {
                    PnnControlBuffer.Add(new PoseData(uPnn.AsArray(), Time));
                }
                if (flagMBPO)
                {
                    MbpoControlBuffer.Add(new PoseData(uMbpo.AsArray(), Time));
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
            
            // Export buffer data
            string iterDir = Path.Combine(OutputDir, $"iteration_{iterIndex}");
            Directory.CreateDirectory(iterDir);
            string path;
            path = Path.Combine(iterDir, "LineTrackError.csv");
            Csv.WriteCsv<PoseData>(path, ErrorBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackControl.csv");
            Csv.WriteCsv(path, ControlBuffer.Memory.ToList());
            path = Path.Combine(iterDir, "LineTrackPose.csv");
            Csv.WriteCsv(path, PoseBuffer.Memory.ToList());
            Thread.Sleep(1000);

            // Plot iteration data in python
            string savePath = Path.Combine(iterDir, $"iteration_fig_{iterIndex}");
            string[] args = new string[2] { iterDir, savePath };
            args = new string[2] { iterDir, savePath };
            //string scriptPath = Path.Combine(ScriptDir, "ilc_plot.py");
            PythonScripts.RunParallel("iter_plot.py", args: args);
            
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

                // PNN Plot
                Pnn.Plot(iterDir:iterDir);

                // PNN Iteration
                List<string> dataDirs = new List<string>()
                {
                    OutputDir
                };
                Pnn.Iteration(nEpoch: 500, dataDirs: dataDirs);
                
            }

            // MBPO
            if (flagMBPO)
            {
                // Write MBPO control CSV file
                path = Path.Combine(iterDir, "LineTrackMbpoControl.csv");
                Csv.WriteCsv(path, MbpoControlBuffer.Memory.ToList());

                // PNN Plot
                Mbpo.Plot(iterDir:iterDir);

                // MBPO Iteration
                List<string> dataDirs = new List<string>()
                {
                    OutputDir
                };
                Mbpo.Iteration(nEpoch: 200, gradSteps: 500, dataDirs: dataDirs);

                
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

        #endregion

        #region Data Acquisition & Processing

        public Vector<double> GetFanucPose()
        {
            return CreateVector.DenseOfArray(PCDK.GetPoseUF());
        }

        public Vector<double> GetVxCameraPose()
        {
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
            var rotationUF = rotationCameraUF * rotationCamera;
            cameraPosition = rotationCameraUF * cameraPosition;
            var orientation = CreateVector.DenseOfArray(MathLib.FixedAnglesIkine(rotationUF));
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
            return true;
        }
    }

    public class RotationIdentification
    {
        // This class encapsulates the identification of the rotation matrix from C-Track Sensor frame to FANUC user frame.

        [JsonIgnore] public Vector<double>[] x = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] y = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] z = new Vector<double>[2];
        [JsonIgnore] public Matrix<double> Rotation;
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

        public void ReadJson(string dir)
        {
            string path = Path.Combine(dir, fileName);
            string jsonString = File.ReadAllText(path);
            double[] array = JsonSerializer.Deserialize<RotationIdentification>(jsonString).RotationArray;
            Rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, array);
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
