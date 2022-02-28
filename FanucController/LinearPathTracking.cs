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
        public string TopDir;
        public string OutputDir;
        public string ReferenceDir;
        public string ScriptsDir;
        public string ProgramName;
        public DpmWatcher Watcher;
        public System.Timers.Timer Timer;
        public Stopwatch Stopwatch;
        public VXelementsUtility Vx;
        public LinearPath linearPath;
        public RotationIdentification rotationId;
        public Dictionary<string, Vector<double>> poseDict;
        public Matrix<double> Kp;
        private bool started;
        private bool stopped;

        public LinearPathTracking(VXelementsUtility vx, System.Timers.Timer timer, Stopwatch stopwatch, string topDir)
        {
            this.Vx = vx;
            this.Timer = timer;
            this.Stopwatch = stopwatch;
            this.TopDir = topDir;
            this.OutputDir = Path.Combine(topDir, "Output");
            this.ReferenceDir = Path.Combine(topDir, "Reference");
            this.ScriptsDir = Path.Combine(topDir, "Scripts");
        }

        // ------------------------------- Actions

        public void Init(string pathPath)
        {
            this.linearPath = LinearPath.ReadJson(pathPath);

        }

        public void Start()
        {
            this.Timer.Start();
            PCDK.Run(this.ProgramName);
        }

        public void Stop()
        {
            this.Timer.Stop();
        }

        public bool Check()
        {
            return true;
        }

        public void RecordPose(string key, Vector<double> pose)
        {
            /* Rotation keys: idX1, idX2, idY1, idY2, idZ1, idZ2, testP1, testP2
             * Linear path keys: startP, endP    
             */
            this.poseDict[key] = pose;
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
                    this.rotationId.x[0] = this.poseDict["idX1"]; this.rotationId.x[1] = this.poseDict["idX2"];
                    this.rotationId.y[0] = this.poseDict["idY1"]; this.rotationId.x[1] = this.poseDict["idY2"];
                    this.rotationId.CalculateRotationMatrixXY();
                    this.rotationId.WriteJson(this.ReferenceDir);
                    break;
                case "xz":
                    this.rotationId.x[0] = this.poseDict["idX1"]; this.rotationId.x[1] = this.poseDict["idX2"];
                    this.rotationId.z[0] = this.poseDict["idZ1"]; this.rotationId.z[1] = this.poseDict["idZ2"];
                    this.rotationId.CalculateRotationMatrixXZ();
                    this.rotationId.WriteJson(this.ReferenceDir);
                    break;
                case "yz":
                    this.rotationId.y[0] = this.poseDict["idY1"]; this.rotationId.y[1] = this.poseDict["idY2"];
                    this.rotationId.z[0] = this.poseDict["idZ1"]; this.rotationId.z[1] = this.poseDict["idZ2"];
                    this.rotationId.CalculateRotationMatrixYZ();
                    this.rotationId.WriteJson(this.ReferenceDir);
                    break;
            }
            
        }


        // ------------------------------- EventHandlers

        private void elapsedEventHandler()
        {
            Vector<double> x = GetVxUFPose();
            Vector<double> closestP = MathLib.PointToLinePoint(linearPath.PointStart, linearPath.PointEnd, x.SubVector(0, 3));
            Vector<double> xd = (closestP.ToColumnMatrix().Stack(x.SubVector(3, 3).ToColumnMatrix())).Column(0);
            Vector<double> e = xd - x;
            Vector<double> u = Pid(e);
            if (this.Watcher.LimitCheck(e, u))
            {
                PCDK.ApplyDPM(u.AsArray());
            }
            // Save data to buffer

        }

        // ------------------------------- Control

        public Vector<double> Pid(Vector<double> error)
        {
            return this.Kp * error;
        }

        // ------------------------------- Data acquisition & processing

        public Vector<double> GetFanucPose()
        {
            return CreateVector.DenseOfArray(PCDK.GetPoseUF());
        }

        public Vector<double> GetVxCameraPose()
        {
            return this.Vx.PoseCameraFrameRkf;
        }

        public Vector<double> GetVxCameraPose(int sampleNum)
        {
            Vector<double> poseSum = CreateVector.Dense<double>(6);
            Vector<double> pose;
            for (int i = 0; i < sampleNum; i++)
            {
                pose = this.Vx.PoseCameraFrameRkf;
                poseSum = CreateVector.DenseOfEnumerable(poseSum.Zip(pose, (x, y) => x + y));
                System.Threading.Thread.Sleep(50);
            }
            poseSum = CreateVector.DenseOfEnumerable(poseSum.Select(x => x / sampleNum));
            return poseSum;
        }

        public Vector<double> VxCameraToUF(Vector<double> cameraPose, Matrix<double> rotationCameraUF)
        {
            var cameraPosition = CreateVector.DenseOfEnumerable(cameraPose.Skip(3));
            var rotationCamera = MathLib.RotationMatrix(cameraPose[4], cameraPose[5], cameraPose[6]);
            var rotationUF = rotationCameraUF * rotationCamera;
            cameraPosition = rotationCameraUF * cameraPosition;
            var orientation = CreateVector.DenseOfArray(MathLib.FixedAnglesIkine(rotationUF));
            return (cameraPosition.ToColumnMatrix().Stack(orientation.ToColumnMatrix())).Column(0);
        }

        public Vector<double> GetVxUFPose()
        {
            var cameraPose = VxCameraToUF(GetVxCameraPose(), this.rotationId.Rotation);
            return cameraPose;
        }

        public Vector<double> GetVxUFPose(int sampleNum)
        {
            var cameraPose = VxCameraToUF(GetVxCameraPose(sampleNum), this.rotationId.Rotation);
            return cameraPose;
        }

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
            this.CumulativeError.Clear();
            this.CumulativeOffset.Clear();
        }

        public bool LimitCheck(Vector<double> error, Vector<double> offset)
        {
            this.CumulativeError += error;
            this.CumulativeOffset += offset;
            if (error.L2Norm() > this.ErrorLimit || this.CumulativeError.L2Norm() > this.CumulativeErrorLimit ||
                offset.L2Norm() > this.OffsetLimit || this.CumulativeOffset.L2Norm() > this.CumulativeOffsetLimit)
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
            var i = (this.x[2] - this.x[1]).SubVector(0, 3);
            var j = (this.y[2] - this.y[1]).SubVector(0, 3);
            var k = MathLib.CrossProduct(i, j);
            this.Rotation = MathLib.RotationMatrix(i, j, k);
        }

        public void CalculateRotationMatrixXZ()
        {
            var i = (this.x[2] - this.x[1]).SubVector(0, 3);
            var k = (this.z[2] - this.z[1]).SubVector(0, 3);
            var j = MathLib.CrossProduct(k, i);
            this.Rotation = MathLib.RotationMatrix(i, j, k);
        }

        public void CalculateRotationMatrixYZ()
        {
            var j = (this.y[2] - this.y[1]).SubVector(0, 3);
            var k = (this.z[2] - this.z[1]).SubVector(0, 3);
            var i = MathLib.CrossProduct(j, k);
            this.Rotation = MathLib.RotationMatrix(i, j, k);
        }

        public double[] RotationArray
        {
            get { return Rotation.ToColumnMajorArray(); }
            set { Rotation = CreateMatrix.DenseOfColumnMajor<double>(3, 3, value); }
        }

        public RotationIdentification ReadJson(string dir)
        {
            string path = Path.Combine(dir, this.fileName);
            return JsonSerializer.Deserialize<RotationIdentification>(path);
        }

        public void WriteJson(string dir)
        {
            string path = Path.Combine(dir, this.fileName);
            string stringJson = JsonSerializer.Serialize(this);
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

        public static LinearPath ReadJson(string path)
        {
            return JsonSerializer.Deserialize<LinearPath>(path);
        }

        public void WriteJson(string dir)
        {
            string path = Path.Combine(dir, this.fileName);
            string stringJson = JsonSerializer.Serialize(this);
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
            string stringJson = JsonSerializer.Serialize(arrayJson);
            File.WriteAllText(path, stringJson);
        }

    }

}
