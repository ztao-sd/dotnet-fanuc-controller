using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FanucController
{
    class OnnxClass
    {
        public static void TestOnnx()
        {
            var watch = new System.Diagnostics.Stopwatch();
            string model_path = @"D:\LocalRepos\mbpo-td3-fanuc-control\results\220205_001_mbpo[100, 80]_td3[60, 50]-[120, 100]_pendulumv0\onnx_models\best_actor_arch_[60, 50].onnx";
            var ort_session = new InferenceSession(model_path);
            var input_tensor = new DenseTensor<float>(new[] { 3 });
            var input = new List<NamedOnnxValue>();
            input.Add(NamedOnnxValue.CreateFromTensor("state", input_tensor));

            watch.Start();
            var output = ort_session.Run(input);
            watch.Stop();

            Console.WriteLine($"Execution time: {watch.ElapsedMilliseconds}");
            var x = output.ToArray();
            Console.WriteLine(x[0].AsTensor<float>()[0]);

        }

    }
}
