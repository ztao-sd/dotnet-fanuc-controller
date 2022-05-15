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
        public int IterCount;
        public int WarmupIters;
        public int WarmpupCount;
        public int TrainingIters;
        public int TrainingCount;
        public int ExplorationIters;
        public int ExplorationCount;
        public int GradientSteps;
        public int NEpochs;
        public int EvalInterval;
        public int ModelUsageIters;

        // Normalization
        private double[] minControl;
        private double[] maxControl;
        private double[] minState;
        private double[] maxState;

        // Data
        public List<string> DataDirs = new List<string>();

        // Control
        public Vector<double> MbpoControl;

        // Reinforcement Learning
        public double[] ExplorationNoise;
        public double WarmupNoise;
        public OrnsteinUlhenbeckNoise OUExplorationNoise;
        public OrnsteinUlhenbeckNoise OUWarmupNoise;

        // Plot
        public string IterFigure = "mbpo_iter_data.png";

        #region Constructors

        public LinearPathTrackingMBPO()
        {
            ScriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonMBPO";
            InputDim = 6;
            OutputDim = 3;
            InputName = "state";
            OutputName = "control";

            EvalInterval = 1;
            WarmupIters = 0;
            NEpochs = 700;
            GradientSteps = 1000;
            TrainingIters = 0;
            ExplorationIters = 0;
            ModelUsageIters = 30;

            // Hyperparameters
            //EvalInterval = 1000;
            //WarmupIters = 1;
            //NEpochs = 5;
            //GradientSteps = 800;
            //TrainingIters = 25;
            //ExplorationIters = 20;
            //ModelUsageIters = 99;
            ExplorationNoise = new double[3] { 0.001, 0.001, 0.001};
            OUExplorationNoise = new OrnsteinUlhenbeckNoise(ExplorationNoise, new double[] { 0.2, 0.2, 0.2 });
            WarmupNoise = 0.10; // ratio of min/max control
            OUWarmupNoise = new OrnsteinUlhenbeckNoise(new double[] {0.006, 0.006, 0.006}, 
                new double[] { 0.2, 0.2, 0.2 });
            minControl = new double[3] { -0.005, -0.005, -0.005 };
            maxControl = new double[3] { 0.005, 0.005, 0.005 };
            minState = new double[6] { -2000, -1200, -200, -0.50, -0.50, -0.50 };
            maxState = new double[6] { -1600, -400, 0, 0.50, 0.50, 0.50 };
        }
        
        #endregion

        #region Actions

        public void Init()
        {
            ActorPath = Path.Combine(OutputDir, "mbpo", "actor.onnx");
            MbpoDir = Path.GetDirectoryName(ActorPath);
            Directory.CreateDirectory(MbpoDir);
            WarmpupCount = 0;
            TrainingCount = 0;
            IterCount = 0;
        }

        public void Reset()
        {
            if (WarmpupCount > WarmupIters)
            {
                NewSession();
            }
            MbpoControl = CreateVector.Dense<double>(6);
            OUExplorationNoise.Reset();
            OUWarmupNoise.Reset();
        }

        public Vector<double> Control(Vector<double> pose, Vector<double> error, bool rand = true)
        {
            MbpoControl.Clear();
            if (WarmpupCount > WarmupIters)
            {
                double[] input = new double[InputDim];
                pose.SubVector(0,3).AsArray().CopyTo(input, 0);
                error.SubVector(0, 3).AsArray().CopyTo(input, 3);
                var control = Forward(input);
                var temp = CreateVector.DenseOfArray<double>(control);
                temp.CopySubVectorTo(MbpoControl, 0, 0, OutputDim);
                
                if (ExplorationCount < ExplorationIters && (IterCount % EvalInterval != 0 || IterCount == 0)) 
                {
                    // MbpoControl += ExplorationControl();
                    MbpoControl += OUExplorationNoise.Sample();
                }
            }
            else
            {
                if (rand)
                {
                    //var control = new double[OutputDim];
                    //for (int i = 0; i < OutputDim; i++)
                    //{
                    //    control[i] = ContinuousUniform.Sample(WarmupNoise * minControl[i], WarmupNoise * maxControl[i]);
                    //}
                    //var temp = CreateVector.DenseOfArray<double>(control);
                    //temp.CopySubVectorTo(MbpoControl, 0, 0, OutputDim);
                    MbpoControl = OUWarmupNoise.Sample();
                }
            }

            return MbpoControl;

        }

        public void Iteration(int nEpoch = 200, int gradSteps = 500, string iterDir=null, List<string> dataDirs = null)
        {

            int warmupInt = (WarmpupCount < WarmupIters) ? 1 : 0;
            int modelUsageInt = (IterCount >= ModelUsageIters) ? 1 : 0;

            // Increment iter count
            if (IterCount % EvalInterval == 0 && IterCount != 0)
            {
                IterCount++;
                return;
            }
            IterCount++;

            

            // Increment warmup count
            WarmpupCount++;
            
            // Increment training count
            TrainingCount++;

            // Increment exploration count
            ExplorationCount++;
            

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
            args.Insert(0, ScriptDir);
            args.Insert(0, iterDir);
            args.Insert(0, MbpoDir);
            args.Insert(0, MbpoDir);
            args.Insert(0, modelUsageInt.ToString());
            args.Insert(0, gradSteps.ToString());
            args.Insert(0, warmupInt.ToString());
            args.Insert(0, nEpoch.ToString());

            if (TrainingCount < TrainingIters)
            {
                PythonScripts.Run(iterScript, args: args.ToArray(), shell: true, dir: ScriptDir);
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
            PythonScripts.RunParallel(plotScript, args: args.ToArray(), dir: ScriptDir);
        }

        #region Helpers

        public Vector<double> ExplorationControl()
        {
            var explorationControl = CreateVector.Dense<double>(6);
            for (int i = 0; i < 3; i++)
            {
                explorationControl[i] = Normal.Sample(0, ExplorationNoise[i]);
                if (Math.Abs(explorationControl[i]) > ExplorationNoise[i] * 5){
                    explorationControl[i] = ExplorationNoise[i] * 5;
                } 
            }

            return explorationControl;
        }

        #endregion

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

        public double[] NormalizeState(double[] stateArray)
        {
            double[] state = new double[stateArray.Length];
            for (int i = 0; i < stateArray.Length; i++)
            {
                state[i] = 2 / (maxState[i] - minState[i]) * (stateArray[i] - minState[i]) - 1;
            }
            return state;
        }

        public double[] Forward(double[] state)
        {
            state = NormalizeState(state);
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

    public class OrnsteinUlhenbeckNoise
    {
        private Vector<double> x0;
        private Vector<double> xPrev;

        private Vector<double> mu;
        private Vector<double> sigma;
        private Vector<double> theta;
        private double dt;

        public OrnsteinUlhenbeckNoise(double[] explorationNoise, double[] friction)
        {
            const int dim = 3;
            mu = CreateVector.Dense<double>(dim);
            sigma = CreateVector.DenseOfArray(explorationNoise);
            theta = CreateVector.DenseOfArray(friction);
            dt = 0.08;

            xPrev = mu;
        }

        public void Reset()
        {
            xPrev = mu;
        }

        public Vector<double> Sample()
        {
            Vector<double> x = CreateVector.Dense<double>(6);
            Vector<double> normalRand = CreateVector.Dense<double>(3);
            for (int i = 0; i < 3; i++)
            {
                normalRand[i] = Normal.Sample(0, 1);
                if (Math.Abs(normalRand[i]) >  5)
                {
                    normalRand[i] = Math.Sign(normalRand[i]) * 5;
                }
            }
            Vector<double> temp = xPrev + theta.PointwiseMultiply(mu - xPrev) * dt + Math.Sqrt(dt) * sigma.PointwiseMultiply(normalRand);
            temp.CopySubVectorTo(x, 0, 0, 3);

            return x;
        }
    }
}
