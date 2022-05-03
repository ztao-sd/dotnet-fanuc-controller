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
    public class LinearPathTrackingPNN
    {
        // File
        public string OutputDir;

        // Model
        public InferenceSession OrtSesssion;
        public int InputDim;
        public int OutputDim1;
        public int OutputDim2;
        public string InputName;
        public string OutputName;
        public string ModelPath;

        // Script
        public string ScriptDir;
        private string iterScript = "iter_pnn.py";
        private string plotScript = "iter_pnn_plot.py";

        // Training
        public int WarmupIters;
        public int WarmpupCount;

        // P Control
        public Matrix<double> Kp;
        public double MinError;
        public double MaxError;
        public Vector<double> PnnControl;

        // Data
        public List<string> DataDirs = new List<string>();

        // PlotF
        public string IterFigure = "pnn_iter_data.png";

        // Data normalizatin
        private double[] minState = new double[4] { 0, -0.20, -0.20, -0.20 };
        private double[] maxState = new double[4] { 35, 0.20, 0.20, 0.20 };

        public LinearPathTrackingPNN()
        {
            InputDim = 4;
            OutputDim1 = 2;
            OutputDim2 = 3;
            // ModelPath = Path.Combine(OutputDir, "predictor.onnx"),
            InputName = "prev_control";
            OutputName = "error";
            ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";
            WarmupIters = 3;
            Kp = CreateMatrix.DenseDiagonal<double>(3, 0.1);
            MinError = -0.08;
            MaxError = 0.08;
        }

        public LinearPathTrackingPNN(string outputDir, string modelPath, string inputName, string outputName, int inputDim, 
            int outputDim1, int outputDim2, string scriptDir)
        {
            OutputDir = outputDir;
            InputDim = inputDim;
            OutputDim1 = outputDim1;
            OutputDim2 = outputDim2;
            ModelPath = modelPath; 
            InputName = inputName;
            OutputName = outputName;
            ScriptDir = scriptDir;
            OrtSesssion = new InferenceSession(modelPath);
        }

        #region Actions

        public void Init()
        { 
            ModelPath = Path.Combine(OutputDir, "pnn_model", "predictor.onnx");
            string ModelDir = Path.GetDirectoryName(ModelPath);
            Directory.CreateDirectory(ModelDir);
            WarmpupCount = 0;
        }

        public void Reset()
        {
            if (WarmpupCount >= WarmupIters)
            {
                NewSession();
            }
            PnnControl = CreateVector.Dense<double>(6);
        }

        public Vector<double> Control(Vector<double> pControl, double time, bool rand=true)
        {
            PnnControl.Clear();
            if (WarmpupCount >= WarmupIters)
            {
                if (time < 2.5)
                {
                    return PnnControl;
                }
                if (pControl.SubVector(0,3).L2Norm() > 0.10 * 0.3)
                {
                    return PnnControl;
                }
                double[] input = new double[InputDim];
                input[0] = time;
                pControl.SubVector(0, 3).AsArray().CopyTo(input, 1);
                var control = Forward(input);
                var vPnn = Kp * CreateVector.DenseOfArray<double>(control);
                vPnn.CopySubVectorTo(PnnControl, 0, 0, 3);
            }
            else
            {
                if (rand)
                {
                    var vPnn = Kp * CreateVector.DenseOfArray<double>(Sample());
                    //PnnControl = CreateVector.Dense<double>(6);
                    vPnn.CopySubVectorTo(PnnControl, 0, 0, 3);
                }
            }

            return PnnControl;
        }

        public void Iteration(int nEpoch = 1000, List<string> dataDirs = null)
        {
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
            args.Insert(0, ModelPath);
            args.Insert(0, nEpoch.ToString());
            if (WarmpupCount < WarmupIters)
            {
                PythonScripts.Run(iterScript, args: args.ToArray(), shell: true);
            }
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
            PythonScripts.RunParallel(plotScript, args: args.ToArray());
        }

        #endregion

        #region ONNX

        public void NewSession()
        {
            OrtSesssion = new InferenceSession(ModelPath);
        }

        public double[] NormalizeControl(double[] controlArray)
        {
            double[] control = new double[controlArray.Length];
            for (int i = 0; i < controlArray.Length; i++)
            {
                control[i] = 2 / (maxState[i] - minState[i]) * (controlArray[i] - minState[i]) - 1;
            }
            return control;
        }

        public double[] Forward(double[] controlArray)
        {
            controlArray = NormalizeControl(controlArray);
            var inputTensor = new DenseTensor<float>(new[] { InputDim });
            for (int i = 0; i < InputDim; i++)
            {
                inputTensor[i] = (float)controlArray[i];
            }
            var input = new List<NamedOnnxValue>();
            input.Add(NamedOnnxValue.CreateFromTensor(InputName, inputTensor));
            var output = OrtSesssion.Run(input).ToArray();
            var result = new double[OutputDim2];
            for (int j = 0; j < OutputDim2; j++)
            {
                result[j] = output[0].AsTensor<float>()[j];
            }

            return result;
        }

        public double[] Sample()
        {
            var error = new double[OutputDim2];
            for (int i = 0; i < OutputDim2; i++)
            {
                error[i] = ContinuousUniform.Sample(MinError, MaxError);
            }
            return error;
        }

        #endregion

        #region Gaussian

        public double[,] ForwardGaussian(double[] controlArray)
        {
            var inputTensor = new DenseTensor<float>(new[] { InputDim });
            for (int i = 0; i < InputDim; i++)
            {
                inputTensor[i] = (float)controlArray[i];
            }
            var input = new List<NamedOnnxValue>();
            input.Add(NamedOnnxValue.CreateFromTensor(InputName, inputTensor));
            var output = OrtSesssion.Run(input).ToArray();
            var result = new double[OutputDim1, OutputDim2];
            for (int i = 0; i < OutputDim1; i++)
            {
                for (int j = 0; j < OutputDim2; j++)
                {
                    result[i, j] = output[i].AsTensor<float>()[j];
                }
            }
            return result;
        }

        public double[] Sample(double[,] normal)
        {
            var error = new double[OutputDim2];
            for (int i = 0; i < normal.GetLength(1); i++)
            {
                error[i] = Normal.Sample(normal[0, i], normal[1, i]);
            }
            return error;
        }

        #endregion

        #region Test

        public static void Test()
        {
            string outputDir = @"D:\Fanuc Experiments\stuff\ML - Python\data\dec_14\run_2";
            string modelPath = @"D:\Fanuc Experiments\stuff\ML - Python\data\dec_14\run_2\mlpg_0401.onnx";
            string scriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";
            var model = new LinearPathTrackingPNN(outputDir:outputDir, modelPath: modelPath, inputName:"input.1", outputName:"error",
                inputDim: 4, outputDim1: 2, outputDim2: 3, scriptDir:scriptDir);

            var normalDistribution = new Normal(0.0, 0.5);
            double[] control = new double[model.InputDim];
            normalDistribution.Samples(control);
            control[0] = 10;

            var error = model.Forward(control);
            Console.WriteLine($"Control: {string.Join(",", control)}");

            //var error = new double[model.OutputDim2];
            //for (int i = 0; i < normal.GetLength(1); i++)
            //{
            //    error[i] = Normal.Sample(normal[0,i], normal[1,i]);
            //}
            Console.WriteLine($"Error: {string.Join(",", error)}");
        }

        #endregion

    }
}
