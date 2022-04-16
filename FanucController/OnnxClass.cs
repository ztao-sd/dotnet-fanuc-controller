using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

namespace FanucController
{
    public class OnnxModel
    {
        public InferenceSession ortSession;
        public string modelPath;
        public int stateDim;
        public int actionDim;
        public string inputName;
        public string outputName;

        public OnnxModel(string modelPath, string inputName="state", string outputName="action",int stateDim=3, int actionDim=1)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.modelPath = modelPath;
            this.inputName = inputName;
            this.outputName = outputName;
            this.ortSession = new InferenceSession(this.modelPath);

        }

        public double[] Forward(double[] stateArray)
        {
          
            var inputTensor = new DenseTensor<float>(new[] { this.stateDim });
            for (int i=0; i<this.stateDim; i++)
            {
                inputTensor[i] = (float) stateArray[i];
            }
            var input = new List<NamedOnnxValue>();
            input.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
            
            var output = this.ortSession.Run(input).ToArray();
            
            double[] result = new double[output.Length];
            int idx = 0;
            foreach (var v in output)
            {
                var testTensor = v.AsTensor<float>();
                var x = (double)testTensor[0];
                result[idx]=v.AsTensor<float>()[0];
                idx++;
            }
            return result;
        }
         
        public static bool Test()
        {
            string modelPath = @"D:\LocalRepos\dotnet-fanuc-controller\OnnxModels\best_actor_arch_[60, 50].onnx";
            var testModel = new OnnxModel(modelPath);
            var watch = new System.Diagnostics.Stopwatch();

            var normalDistribution = new Normal(0.0, 0.5);
            double[] normalArray = new double[testModel.stateDim];
            normalDistribution.Samples(normalArray);

            double[] actionArray;
            watch.Start();
            actionArray = testModel.Forward(normalArray);
            watch.Stop();

            Console.WriteLine($"Action: {actionArray[0]}");
            Console.WriteLine($"Length: {actionArray.Length}");
            Console.WriteLine($"Execution time: {watch.ElapsedMilliseconds}");

            return true;

        }

        public static void Test2()
        {
            string modelPath = @"D:\Fanuc Experiments\stuff\ML - Python\data\dec_14\run_2\mlpg_0401.onnx";
            var model = new OnnxModel(modelPath, inputName:"input.1", stateDim:4, actionDim:3);

            var normalDistribution = new Normal(0.0, 0.5);
            double[] control = new double[model.stateDim];
            normalDistribution.Samples(control);

            var error = model.Forward(control);
            Console.WriteLine($"Control: {control}");
            Console.WriteLine($"Error: {error}");
        }
    }
}
