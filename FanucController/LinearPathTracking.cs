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
        public string SaveDir;
        public string ReadDir;
        public string ProgramName;
        public System.Timers.Timer Timer;
        public Stopwatch Stopwatch;
        public VXelementsUtility Vx;
        public LinearPath linearPath;
        public RotationIdentification rotationId;
        public Matrix<double> Kp;
        private bool started;
        private bool stopped;

        public LinearPathTracking(VXelementsUtility vx, System.Timers.Timer timer, Stopwatch stopwatch)
        {
            this.Vx = vx;
            this.Timer = timer;
            this.Stopwatch = stopwatch;
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

        public void RecordIdentificationPose(ref Vector<double> idPose, int sampleNum)
        {
            idPose = GetVxCameraPose(sampleNum);
        }

        public void RotationIdentify()
        {

        }


        // ------------------------------- EventHandlers

        private void elapsedEventHandler()
        { 

        }

        // ------------------------------- Control

        public Vector<double> Pid(Vector<double> error)
        {
            return this.Kp * error;
        }

        // ------------------------------- Data acquisition & processing

        public Vector<double> GetFanucPose()
        {
            return CreateVector.DenseOfArray(PCDK.GetPose());
        }

        public Vector<double> GetVxCameraPose()
        {
            double[] pose = new double[6];
            pose[0] = this.Vx.translationCameraFrame.X;
            pose[1] = this.Vx.translationCameraFrame.Y;
            pose[2] = this.Vx.translationCameraFrame.Z;
            pose[3] = this.Vx.rotationCameraFrame.X;
            pose[4] = this.Vx.rotationCameraFrame.Y;
            pose[5] = this.Vx.rotationCameraFrame.Z;
            return CreateVector.DenseOfArray(pose);
        }

        public Vector<double> GetVxCameraPose(int sampleNum)
        {
            double[] poseSum = new double[6];
            double[] pose = new double[6];
            for (int i = 0; i < sampleNum; i++)
            {
                pose[0] = this.Vx.translationCameraFrame.X;
                pose[1] = this.Vx.translationCameraFrame.Y;
                pose[2] = this.Vx.translationCameraFrame.Z;
                pose[3] = this.Vx.rotationCameraFrame.X;
                pose[4] = this.Vx.rotationCameraFrame.Y;
                pose[5] = this.Vx.rotationCameraFrame.Z;
                poseSum = poseSum.Zip(pose, (x, y) => x + y).ToArray();
                System.Threading.Thread.Sleep(50);
            }
            poseSum = poseSum.Select(x => x / sampleNum).ToArray();
            return CreateVector.DenseOfArray(poseSum);
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

    }

    public class DpmWatcher
    {
        public double OffsetLimit;
        public double CumulativeOffset;
        public double CumulativeOffsetLimit;
        public double ErrorLimit;
        public double CumulativeError;
        public double CumulativeErrorLimit;

        public void Clear()
        {
            this.CumulativeError = 0;
            this.CumulativeOffset = 0;
        }
    }

    public class RotationIdentification
    {
        [JsonIgnore] public Vector<double>[] x = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] y = new Vector<double>[2];
        [JsonIgnore] public Vector<double>[] z = new Vector<double>[2];
        [JsonIgnore] public Matrix<double> Rotation;

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

        public static RotationIdentification ReadJson(string path)
        {
            return JsonSerializer.Deserialize<RotationIdentification>(path);
        }

        public void WriteJson(string path)
        {
            string stringJson = JsonSerializer.Serialize(this);
            File.WriteAllText(path, stringJson);
        }

    }

    public class LinearPath
    {
        [JsonIgnore]
        public Matrix<double> Rotation; //Rotation Camera -> UF
        [JsonIgnore]
        public Vector<double> PointStart;
        [JsonIgnore]
        public Vector<double> PointEnd;

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

        public void WriteJson(string path)
        {
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
