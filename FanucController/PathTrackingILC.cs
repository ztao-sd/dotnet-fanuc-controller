using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
using MathNet.Numerics.LinearAlgebra;
using FanucPCDK;
using Serilog;

namespace FanucController
{
    public class IterativeLearningControl
    {
        // Raw data
        public List<Vector<double>> Errors = new List<Vector<double>>();
        public List<double> Times = new List<double>();
        public double StartTime = 0;

        // Processed data
        public int Length; 
        public Vector<double>[] ControlArray;
        public double[] TimeArray;
        public Vector<double>[] NextControlArray;
        public double[] NextTimeArray;
        public Vector<double>[] ErrorArray;
        public Vector<double>[] ErrorDotArray;

        // ILC Parameters
        public Matrix<double> Kp;
        public Matrix<double> Kd;

        // Python Script
        public string ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";

        public IterativeLearningControl()
        {

        }

        public IterativeLearningControl(Matrix<double> kp, Matrix<double> kd, int length = 2000)
        {
            Length = length;
            ControlArray = new Vector<double>[length];
            TimeArray = new double[Length];
            NextControlArray = new Vector<double>[Length];
            NextTimeArray = new double[Length];
            ErrorArray = new Vector<double>[Length];
            ErrorDotArray = new Vector<double>[Length];

            Kp = kp;
            Kd = kd;
        }

        public void Init(double endTime = 100)
        {
            Reset();
            TimeArray = MathLib.LinSpace(0, endTime, Length);
            ControlArray = ControlArray.Select(u => CreateVector.Dense<double>(6)).ToArray();
        }

        public void Init(IterativeLearningControl lastIlc)
        {
            Reset();
            TimeArray = lastIlc.NextTimeArray;
            ControlArray = lastIlc.NextControlArray;
        }
        

        public void Add(Vector<double> error, double time)
        {
            Errors.Add(error);
            Times.Add(time);
        }

        public void Reset()
        {
            Errors.Clear();
            Times.Clear();
            StartTime = 0;
            Array.Clear(ControlArray, 0, ControlArray.Length);
            Array.Clear(TimeArray, 0, TimeArray.Length);
            Array.Clear(NextControlArray, 0, NextControlArray.Length);
            Array.Clear(NextTimeArray, 0, NextTimeArray.Length);
            Array.Clear(ErrorArray, 0, ErrorArray.Length);
            Array.Clear(ErrorDotArray, 0, ErrorDotArray.Length);
        }

        public void ShiftTime(double? startTime=null)
        {
            if (startTime == null)
            {
                StartTime = Times[0];
            }else
            {
                StartTime = startTime.Value;
            }
            Times = Times.Select(t => t - StartTime).ToList();
        }

        #region Next Control Array Computation

        public void PControl()
        {
            Vector<double> control;
            Vector<double> nextControl;
            Vector<double> error;

            // Interpolate
            NextTimeArray = MathLib.LinSpace(Times.First(), Times.Last(), Length);
            ErrorArray = MathLib.InterpV(Errors.ToArray(), Times.ToArray(), NextTimeArray);
            ControlArray = MathLib.InterpV(ControlArray, TimeArray, NextTimeArray);

            for (int i = 0; i < NextControlArray.Length; i++)
            {
                control = ControlArray[i];
                error = ErrorArray[i];
                nextControl = control + Kp.Multiply(error);
                NextControlArray[i] = nextControl;
            }
        }

        public void PDControl()
        {
            
            Vector<double> control;
            Vector<double> nextControl;
            Vector<double> error;
            Vector<double> errorDot;

            // Interpolate
            NextTimeArray = MathLib.LinSpace(Times.First(), Times.Last(), Length);
            ErrorArray = MathLib.InterpV(Errors.ToArray(), Times.ToArray(), NextTimeArray);
            ControlArray = MathLib.InterpV(ControlArray, TimeArray, NextTimeArray);
            ErrorDotArray = ErrorArray.Skip(1).Append(CreateVector.Dense<double>(6)).Zip(ErrorArray, (e1, e0) => e1 - e0).ToArray();

            for (int i = 0; i < NextControlArray.Length; i++)
            {
                control = ControlArray[i];
                error = ErrorArray[i];
                errorDot = ErrorDotArray[i];
                nextControl = control + Kp.Multiply(error) + Kd.Multiply(errorDot);
                NextControlArray[i] = nextControl;
            }
        }

        #endregion

        #region Lookup Table

        public Vector<double> Query(double qTime)
        {
            return MathLib.InterpV(ControlArray, TimeArray, qTime);
        }

        #endregion

        #region Import/Export CSV

        public void ToCsv(string path)
        {
            List<PoseData> dataList = new List<PoseData>();
            for (int i = 0; i < NextControlArray.Length; i++)
            {
                dataList.Add(new PoseData(NextControlArray[i].AsArray(), NextTimeArray[i]));
            }
            Csv.WriteCsv(path, dataList);
        }

        public void FromCsv(string controlPath, string errorPath=null)
        {
            List<PoseData> dataList = Csv.ReadCsv<PoseData>(controlPath);
            ControlArray = new Vector<double>[dataList.Count];
            TimeArray = new double[dataList.Count];
            for (int i = 0; i < dataList.Count; i++)
            {
                var poseData = dataList[i];
                var vector = poseData.ToVector();
                ControlArray[i] = vector;
                TimeArray[i] = poseData.Time;
            }

            if (errorPath != null)
            {
                dataList = Csv.ReadCsv<PoseData>(errorPath);
                Errors.Clear();
                Times.Clear();
                for (int i = 0; i < dataList.Count; i++)
                {
                    var poseData = dataList[i];
                    var vector = poseData.ToVector();
                    Errors.Add(vector);
                    Times.Add(poseData.Time);
                }
            }
        }

        #endregion

        #region Python Scripts

        public void Iteration(string savePath, string iterDir)
        {
            string[] args = new string[2] { iterDir, savePath };
            string scriptPath = Path.Combine(ScriptDir, "iter_ilc_plot.py");
            PythonScripts.RunParallel("iter_ilc_plot.py", args: args);
        }
        

        #endregion

    }
}
