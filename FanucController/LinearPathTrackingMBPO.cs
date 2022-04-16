using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
        public int OutputDim1;
        public int OutputDim2;
        public string InputName;
        public string OutputName;
        public string ModelPath;

        // Script
        public string ScriptDir;
        private string iterScript = "iter_mbpo.py";
        private string plotScript = "iter_mbpo_plot.py";

        // Training
        public int WarmupIters;
        public int WarmpupCount;

        // P Control
        public Matrix<double> Kp;
        public double MinError;
        public double MaxError;

        // Data
        public List<string> DataDirs = new List<string>();

        // Plot
        public string IterFigure = "mbpo_iter_data.png";

        public LinearPathTrackingMBPO()
        {

        }

        public void NewSession()
        {
            OrtSesssion = new InferenceSession(ModelPath);
        }

        public double[] Forward(double[] state)
        {
            var inputTensor = new DenseTensor<float>(new[] { InputDim });
            for (int i = 0; i < InputDim; i++)
            {
                inputTensor[i] = (float) state[i];
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

    }
}
