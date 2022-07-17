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
    public class PathTrackingPNN
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
        public int NEpochs;
        public int WarmupIters;
        public int WarmpupCount;
        public int TrainingIters;
        public int IterCount;
        public OrnsteinUlhenbeckNoise OUWarmupNoise;

        // P Control
        public Matrix<double> Kp;
        public double MinError;
        public double MaxError;
        public Vector<double> PnnControl;
        public Vector<double> PnnError;

        // PID Control
        public PathTrackingPID Pid;

        // Data
        public List<string> DataDirs = new List<string>();

        // PlotF
        public string IterFigure = "pnn_iter_data.png";

        // Data normalizatin
        private double[] minState = new double[4] { 0, -0.20, -0.20, -0.20 };
        private double[] maxState = new double[4] { 35, 0.20, 0.20, 0.20 };

        public PathTrackingPNN()
        {
            InputDim = 6;
            OutputDim1 = 2;
            OutputDim2 = 3;
            // ModelPath = Path.Combine(OutputDir, "predictor.onnx"),
            InputName = "prev_control";
            OutputName = "error";
            ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";
            
            MinError = -0.08;
            MaxError = 0.08;
            minState = new double[6] { -2000, -1200, -200, -0.005, -0.005, -0.005 };
            maxState = new double[6] { -1600, -400, 0, 0.005, 0.005, 0.005 };

            // P
            Kp = CreateMatrix.DenseDiagonal<double>(3, 0.1);

            // Training
            NEpochs = 400;
            WarmupIters = 0;
            TrainingIters = 0;
            OUWarmupNoise = new OrnsteinUlhenbeckNoise(new double[] { 0.005, 0.005, 0.005 },
                new double[] { 0.2, 0.2, 0.2 });

            // PID 
            Pid = new PathTrackingPID();
            Pid.Dim = 3;
            Pid.Kp = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.05, 0.00, 0.00 });
            Pid.Ki = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.01, 0.0005, 0.0002 });
            Pid.Kd = CreateMatrix.DenseOfDiagonalArray(new double[3] { 0.10, 0.10, 0.10 });
            Pid.Errors = new List<Vector<double>>();
            Pid.ControlSignal = CreateVector.Dense<double>(Pid.Dim);
        }

        public PathTrackingPNN(string outputDir, string modelPath, string inputName, string outputName, int inputDim, 
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
            IterCount = 0;

            // PID
            Pid.Init();
        }

        public void Reset()
        {
            if (WarmpupCount >= WarmupIters)
            {
                NewSession();
            }
            PnnControl = CreateVector.Dense<double>(6);
            PnnError = CreateVector.Dense<double>(6);

            // PID
            Pid.Reset();
            OUWarmupNoise.Reset();
        }

        public Vector<double> Control(Vector<double> pose, Vector<double> pidControl, double time, bool rand=true)
        {
            PnnControl.Clear();
            PnnError.Clear();

            //if (IterCount > TrainingIters)
            //{
            //    return PnnControl;
            //}

            if (WarmpupCount >= WarmupIters)
            {
                if (time < 0)
                {
                    return PnnControl;
                }
                if (pidControl.SubVector(0,3).L2Norm() > 0.10 * 1)
                {
                    return PnnControl;
                }
                double[] input = new double[InputDim];
                pose.SubVector(0, 3).AsArray().CopyTo(input, 0);
                pidControl.SubVector(0, 3).AsArray().CopyTo(input, 3);
                var error = CreateVector.DenseOfArray<double>(Forward(input));
                error.CopySubVectorTo(PnnError, 0, 0, 3);
                PnnControl = Pid.Control(error);
                //vPnn.CopySubVectorTo(PnnControl, 0, 0, 3);
            }
            else
            {
                if (rand)
                {
                    PnnControl = OUWarmupNoise.Sample();
                    //PnnControl = CreateVector.Dense<double>(6);
                    //vPnn.CopySubVectorTo(PnnControl, 0, 0, 3);
                }
            }

            return PnnControl;
        }

        public void Iteration(int nEpoch = 1000, List<string> dataDirs = null)
        {
            if (IterCount >= TrainingIters)
            {
                IterCount++;
                return;
            }

            // Increment warmup count
            WarmpupCount++;

            // Increment iter count
            IterCount++;

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
            PythonScripts.Run(iterScript, args: args.ToArray(), shell: true);
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
            var model = new PathTrackingPNN(outputDir:outputDir, modelPath: modelPath, inputName:"input.1", outputName:"error",
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
