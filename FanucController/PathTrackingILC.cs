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
using MathNet.Numerics.LinearAlgebra.Solvers;

namespace FanucController
{
    /// <summary>
    /// Not used.
    /// </summary>
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
            ControlArray = ControlArray.Select(u => u = CreateVector.Dense<double>(6)).ToArray();
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

        public void ShiftTime(double? startTime = null)
        {
            if (startTime == null)
            {
                StartTime = Times[0];
            }
            else
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

        public void FromCsv(string controlPath, string errorPath = null)
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

    public class PidIlc : IPathTracking
    {
        // Constants
        const int dim = 6;
        const int timeSteps = 2000;
        const int histSize = 3;

        // Pid coefficients.
        Matrix<double> kp = CreateMatrix.DenseDiagonal<double>(dim, 0.01);
        Matrix<double> ki = CreateMatrix.DenseDiagonal<double>(dim, 0.01);
        Matrix<double> kd = CreateMatrix.DenseDiagonal<double>(dim, 0.01);

        // ILC parameters.
        int iterCount = 0;
        bool firstLoading = false; // Flag for first loading.
        //double startTime = 0.0;
        //double endTime = 0.0;
        Vector<double> control = CreateVector.Dense<double>(6);

        // Error and time lists
        List<Vector<double>> tempErrors = new List<Vector<double>>();
        List<double> tempTimes = new List<double>();

        // Input Data
        double[] times = new double[timeSteps];
        Vector<double>[] errors = new Vector<double>[timeSteps];
        Vector<double>[] errorDots = new Vector<double>[timeSteps];
        Vector<double>[] errorDotDots = new Vector<double>[timeSteps];
        Vector<double>[] controls = new Vector<double>[timeSteps];

        // Output Data
        double[] nextTimes = new double[timeSteps];
        //Vector<double>[] nextErrors = new Vector<double>[timeSteps];
        Vector<double>[] nextControls = new Vector<double>[timeSteps];

        // Paths and directories
        public string firstIterDir = @"";
        public string iterDir = @"";
        public string ilcControlName = "LineTrackPidIlcControl.csv";
        public string ilcErrorName = "LineTrackError.csv";


        // Python script directory.
        public string ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";

        public PidIlc(string firstIterDir=null)
        {
            this.firstIterDir = firstIterDir;

            // PID Gains
            var pGains = new double[] { 0.01, 0.01, 0.01, 0.0, 0.0, 0.0 };
            var iGains = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var dGains = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (int i = 0; i < dim; i++)
            {
                kp[i, i] = pGains[i];
                ki[i, i] = iGains[i];
                kd[i, i] = dGains[i];
            }
        }

        #region Actions

        public void Init()
        {
            Reset();
            times = MathLib.LinSpace(0.0, 2000, timeSteps);
            for(int i = 0; i < timeSteps; i++)
            {
                controls[i] = CreateVector.Dense<double>(6);
            }
            firstLoading = false;
        }

        public void Init(string firstIterDir)
        {
            // Load control and error data from CSV files.
            Reset();
            if (firstIterDir != null)
            {
                FromCsv(firstIterDir);
            }
            else
            {
                FromCsv(this.firstIterDir);
            }
            firstLoading = true;
        }

        public void Reset()
        {
            // Clear temporary errors and times.
            tempErrors.Clear();
            tempTimes.Clear();
        }

        public Vector<double> Control(Vector<double> error, Vector<double> pose, double time = 0)
        {
            tempErrors.Add(error);
            tempTimes.Add(time);
            control = MathLib.InterpV(controls, times, time);
            return control;
        }

        public void Iteration()
        {

        }

        public void Iteration(string iterDir, bool verbose = false)
        {

            // Resize times, errors and controls.
            nextTimes = MathLib.LinSpace(tempTimes.First(), tempTimes.Last(), timeSteps);
            //nextTimes = MathLib.LinSpace(0.0, 17.0, timeSteps);
            controls = MathLib.InterpV(controls, times, nextTimes);
            if (!firstLoading)
            {
                errors = MathLib.InterpV(tempErrors.ToArray(), tempTimes.ToArray(), nextTimes);
            }
            else if (iterCount == 0)
            {
                errors = MathLib.InterpV(errors, times, nextTimes);
            }

            // Derivative of the errors.
            var tempError = errors.ToList();
            tempError.Insert(0, CreateVector.Dense<double>(6));
            tempError.RemoveAt(tempError.Count - 1);
            errorDots = tempError.Zip(errors, (e1, e0) => e0 - e1).ToArray();
            var tempErrorDot = errorDots.ToList();
            tempErrorDot.Insert(0, CreateVector.Dense<double>(6));
            tempErrorDot.RemoveAt(tempErrorDot.Count - 1);
            errorDotDots = tempErrorDot.Zip(errorDots, (e1, e0) => e0 - e1).ToArray();

            // Compute ILC control.
            var ilcControl = CreateVector.Dense<double>(6);
            for (int i = 0; i < timeSteps; i++)
            {
                ilcControl += kp * errorDots[i] + ki * errors[i] + kd * errorDotDots[i];
                nextControls[i] = controls[i] + ilcControl;
            }

            // Export to Csv files.
            ToCsv(iterDir, verbose);
            System.Threading.Thread.Sleep(50);

            // Update.
            iterCount++;
            Array.Copy(nextControls, controls, timeSteps);
            Array.Copy(nextTimes, times, timeSteps);

        }


        #endregion

        #region I/O

        public void FromCsv(string iterDir)
        {
            // Read control and error data.
            string path = Path.Combine(iterDir, ilcErrorName);
            List<PoseData> errorPoseData = Csv.ReadCsv<PoseData>(path);
            path = Path.Combine(iterDir, ilcControlName);
            List<PoseData> controlPoseData = Csv.ReadCsv<PoseData>(path);

            // Convert to list of vectors
            var errorList = errorPoseData.Select(pose => pose.ToVector()).ToList();
            var controlList = controlPoseData.Select(pose => pose.ToVector()).ToList();
            var timeList = errorPoseData.Select(pose => pose.Time).ToList();

            // Resize list of vectors
            times = MathLib.LinSpace(timeList.First(), timeList.Last(), timeSteps);
            errors = MathLib.InterpV(errorList.ToArray(), timeList.ToArray(), times);
            controls = MathLib.InterpV(controlList.ToArray(), timeList.ToArray(), times);
        }

        public void ToCsv(string iterDir, bool verbose = false)
        {
            List<PoseData> ilcControlList = new List<PoseData>();
            for (int i = 0; i < nextControls.Length; i++)
            {
                ilcControlList.Add(new PoseData(nextControls[i].AsArray(), nextTimes[i]));


            }
            string path = Path.Combine(iterDir, ilcControlName);
            Csv.WriteCsv(path, ilcControlList);

            if (verbose)
            {
                List<PoseData> dataList = new List<PoseData>();

                dataList.Clear();
                for (int i = 0; i < controls.Length; i++)
                {
                    dataList.Add(new PoseData(controls[i].AsArray(), times[i]));
                }
                path = Path.Combine(iterDir, "LineTrackPidIlcPrevControl.csv");
                Csv.WriteCsv(path, dataList);

                dataList.Clear();
                for (int i = 0; i < errors.Length; i++)
                {
                    dataList.Add(new PoseData(errors[i].AsArray(), nextTimes[i]));
                }
                path = Path.Combine(iterDir, "LineTrackPidIlcError.csv");
                Csv.WriteCsv(path, dataList);

                dataList.Clear();
                for (int i = 0; i < errorDots.Length; i++)
                {
                    dataList.Add(new PoseData(errorDots[i].AsArray(), nextTimes[i]));
                }
                path = Path.Combine(iterDir, "LineTrackPidIlcErrorDot.csv");
                Csv.WriteCsv(path, dataList);

            }
        }

        #endregion

        #region Testing

        public void Test()
        {
            string iterDir = @"C:\Users\Tao\LocalRepos\Fanuc Experiments\mbpo\test-0516\eval-2\output\iteration_3";
            Init(firstIterDir: null);
            Iteration(iterDir, true);
        }

        #endregion

    }
}
