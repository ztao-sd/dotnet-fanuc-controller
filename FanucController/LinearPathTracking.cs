using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using FanucPCDK;


namespace FanucController
{
    public class LinearPathTracking
    {
        #region Fields
        
        public string TopDir;
        public string OutputDir;
        public string ReferenceDir;
        public string ScriptsDir;
        public string ProgramName;
        public DpmWatcher Watcher;
        public System.Timers.Timer Timer;
        public Stopwatch Stopwatch;
        public VXelementsUtility Vx;
        public LinearPath LinearPath;
        public RotationIdentification RotationId;
        public Dictionary<string, Vector<double>> PoseDict;
        public Matrix<double> Kp;
        public Buffer<PoseData> ErrorBuffer;
        public Buffer<PoseData> ControlBuffer;
        public Buffer<PoseData> PoseBuffer;
        public double Time;

        protected bool digOut2;
        protected bool digOut3;
        protected bool progStarted;
        protected bool progStopped;
        protected bool iterStarted;
        protected bool iterStopped;
        protected bool dpm;
        protected int iterNum;
        protected int iterIndex;
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

            // P Control
            Kp = CreateMatrix.Dense<double>(6, 6);
            Kp[0,0] = 0.3; Kp[1, 1] = 0.3; Kp[2, 2] = 0.3;

            // Buffers
            ErrorBuffer = new Buffer<PoseData>(20_000);
            ControlBuffer = new Buffer<PoseData>(20_000);
            PoseBuffer = new Buffer<PoseData>(20_000);
        }
        #endregion

        #region Path Tracking Actions

        public void Init(string pathPath, string progName, int iter=1, bool dpm=false)
        {
            // Program & Path Info
            LinearPath = LinearPath.FromJson(pathPath);
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
            PCDK.SetupDPM(1);
            PCDK.SetDigitalOutput(2, false);
            // P Control
            Kp = CreateMatrix.Dense<double>(6, 6);
            Kp[0, 0] = 0.3; Kp[1, 1] = 0.3; Kp[2, 2] = 0.3;
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
            // DPM
            this.dpm = dpm;
        }

        public void Start()
        {
            if (!Timer.Enabled)
            {
                Timer.Start();
            }
            PCDK.Run(this.ProgramName);
            Timer.Elapsed += Loop;
        }

        public void Loop(object s, EventArgs ev)
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
            }
            if (!iterStopped && progStopped)
            {
                digOut2 = false;
                PCDK.SetDigitalOutput(2, digOut2);
                iterStopped = true;
            }
            if (iterStarted && !iterStopped)
            {
                // Calculate error
                Vector<double> x = GetVxUFPose();
                Vector<double> closestP = MathLib.PointToLinePoint(LinearPath.PointStart.SubVector(0,3), LinearPath.PointEnd.SubVector(0,3), x.SubVector(0, 3));
                Vector<double> xd = (closestP.ToColumnMatrix().Stack(x.SubVector(3, 3).ToColumnMatrix())).Column(0);
                Vector<double> e = xd - x;
                // Control
                Vector<double> u = Pid(e);
                double time = Stopwatch.Elapsed.TotalSeconds;
                if (this.Watcher.LimitCheck(e.SubVector(0,3), u.SubVector(0,3)))
                {
                    if (dpm)
                    {
                        PCDK.ApplyDPM(u.AsArray(), 1);
                    }
                }
                // Save data to buffer
                ErrorBuffer.Add(new PoseData(e.AsArray(), time));
                ControlBuffer.Add(new PoseData(u.AsArray(), time));
                PoseBuffer.Add(new PoseData(x.AsArray(), time));
            }
            // Stop program if condition is reached
            if (iterStopped)
            {
                Stop();
            }

        }

        public void Stop()
        {
            // Timer
            //Timer.Stop();
            Timer.Elapsed -= Loop;
            // Reset flag
            progStarted = false;
            progStopped = false;
            iterStarted = false;
            iterStopped = false;
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
            // Restart 
            iterIndex++;
            if (iterIndex < iterNum)
            {
                Console.WriteLine("Continue?");
                Console.ReadLine();
                Start();
            }
        }

        public bool Check()
        {
            return true;
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
    }

    public class DpmWatcher
    {
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
