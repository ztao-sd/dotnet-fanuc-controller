using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{
    public class LinearPathTrackingMBPO
    {
        // File
        public string OutputDir;

        // Model
        public InferenceSession OrtSesssion;
        public int InputDim;
        public int OutputDim;
        public string InputName;
        public string OutputName;
        public string ActorPath;
        public string MbpoDir;

        // Script
        public string ScriptDir;
        private string iterScript = "iter_mbpo.py";
        private string plotScript = "iter_mbpo_plot.py";

        // Training
        public int WarmupIters;
        public int WarmpupCount;
        public int GradientSteps;

        // Normalization
        private double[] minControl;
        private double[] maxControl;

        // Data
        public List<string> DataDirs = new List<string>();

        // Control
        public Vector<double> MbpoControl;

        // Plot
        public string IterFigure = "mbpo_iter_data.png";

        #region Constructors

        public LinearPathTrackingMBPO()
        {
            ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonMBPO";
            minControl = new double[3] { -0.05, -0.05, -0.05 };
            maxControl = new double[3] { 0.05, 0.05, 0.05 };
            WarmupIters = 1;
            InputDim = 4;
            OutputDim = 3;
            InputName = "state";
            OutputName = "control";
        }

        #endregion

        #region Actions

        public void Init()
        {
            ActorPath = Path.Combine(OutputDir, "mbpo", "actor.onnx");
            MbpoDir = Path.GetDirectoryName(ActorPath);
            Directory.CreateDirectory(MbpoDir);
            WarmpupCount = 0;
        }

        public void Reset()
        {
            if (WarmpupCount >= WarmupIters)
            {
                NewSession();
            }
            MbpoControl = CreateVector.Dense<double>(6);
        }

        public Vector<double> Control(Vector<double> error, double time, bool rand = true)
        {
            MbpoControl.Clear();
            if (WarmpupCount >= WarmupIters)
            {
                double[] input = new double[InputDim];
                input[0] = time;
                error.SubVector(0, 3).AsArray().CopyTo(input, 1);
                var control = Forward(input);
                var temp = CreateVector.DenseOfArray<double>(control);
                temp.CopySubVectorTo(MbpoControl, 0, 0, OutputDim);
            }
            else
            {
                if (rand)
                {
                    var control = new double[OutputDim];
                    for (int i = 0; i < OutputDim; i++)
                    {
                        control[i] = ContinuousUniform.Sample(0.3 * minControl[i], 0.3 * maxControl[i]);
                    }
                    var temp = CreateVector.DenseOfArray<double>(control);
                    temp.CopySubVectorTo(MbpoControl, 0, 0, OutputDim);
                }
            }

            return MbpoControl;

        }

        public void Iteration(int nEpoch = 200, int gradSteps = 500, List<string> dataDirs = null)
        {

            int warmupInt = (WarmpupCount < WarmupIters) ? 1 : 0;

            // Increment warmup count
            WarmpupCount++;

            // Train prediction model with Python script
            if (dataDirs is null)
            {
                DataDirs.Clear();
                DataDirs.Add(OutputDir);
            }
            else
            {
                DataDirs.Clear();
                DataDirs = dataDirs;
            }
            string scriptPath = Path.Combine(ScriptDir, iterScript);
            List<string> args = DataDirs;
            args.Insert(0, MbpoDir);
            args.Insert(0, MbpoDir);
            args.Insert(0, gradSteps.ToString());
            args.Insert(0, warmupInt.ToString());
            args.Insert(0, nEpoch.ToString());

            PythonScripts.Run(iterScript, args: args.ToArray(), shell: true, dir: ScriptDir);
        }

        public void Plot(string iterDir, string savePath = null)
        {
            if (savePath is null)
            {
                savePath = Path.Combine(iterDir, IterFigure);
            }
            // string scriptPath = Path.Combine(ScriptDir, plotScript);
            List<string> args = new List<string>()
            {
                iterDir,
                savePath
            };
            PythonScripts.RunParallel(plotScript, args: args.ToArray(), dir: ScriptDir);
        }

        #endregion


        #region ONNX

        public void NewSession()
        {
            OrtSesssion = new InferenceSession(ActorPath);
        }

        public double[] NormalizeControl(double[] controlArray)
        {
            double[] control = new double[controlArray.Length];
            for (int i = 0; i < controlArray.Length; i++)
            {
                control[i] = (maxControl[i] - minControl[i]) / 2 * (controlArray[i] + 1) + minControl[i];
            }
            return control;
        }

        public double[] Forward(double[] state)
        {
            var inputTensor = new DenseTensor<float>(new[] { InputDim });
            for (int i = 0; i < InputDim; i++)
            {
                inputTensor[i] = (float)state[i];
            }
            var input = new List<NamedOnnxValue>();
            input.Add(NamedOnnxValue.CreateFromTensor(InputName, inputTensor));
            var output = OrtSesssion.Run(input).ToArray();
            var result = new double[OutputDim];
            for (int j = 0; j < OutputDim; j++)
            {
                result[j] = output[0].AsTensor<float>()[j];
            }
            result = NormalizeControl(result);
            return result;
        }

        #endregion



    }
}
