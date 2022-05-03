using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{
    public class LinearPathTrackingBPNNPID
    {
        // Memory Buffer
        public Queue<Vector<double>> errorQueue;

        // Back Propagation
        public BPNNPID BpX;
        public BPNNPID BpY;
        public BPNNPID BpZ;

        // Control
        private Vector<double> dError;
        private Vector<double> dErrorDot;
        private Vector<double> dErrorInt;

        // Limits
        public double[] minReference = new double[3] { -2150, -1150, -100 };
        public double[] maxReference = new double[3] { -1900, -400, -20 };
        public double[] minError = new double[3] { -0.20, -0.20, -0.20 };
        public double[] maxError = new double[3] { 0.20, 0.20, 0.20 };

        #region BackProp

        public class BPNNPID
        {

            public BPNNPID()
            {
                //Matrix<double> wi = Matrix<double>.Build.Random(5, 4,new gamma(-0.5,0.5));

                // Input weights
                wi = CreateMatrix.DenseOfArray(new double[,] {
                { 0.3147,-0.4025,-0.3424,-0.3581},
                { 0.4058,-0.2215, 0.4706,-0.0782},
                {-0.3730, 0.0469, 0.4572, 0.4157},
                { 0.4134, 0.4575,-0.0146, 0.2922},
                { 0.1324, 0.4649, 0.3003, 0.4595}});
                wi_1 = wi;
                wi_2 = wi;
                wi_3 = wi;

                // Output weights
                wo = CreateMatrix.DenseOfArray(new double[,] {
                { 0.0438, 0.1314, 0.1183, 0.4981,0.0008},
                { 0.4795,-0.2146, 0.1486,-0.1219,0.4572},
                {-0.3443, 0.4441, 0.2451,-0.4927,-0.3680}});
                wo_1 = wo;
                wo_2 = wo;
                wo_3 = wo;

                error_1 = 0;
                error_2 = 0;
            }

            //Parameters of NN
            private double xite = 0.28; //Learning rate
            private double alp = 0.04; //Inertia coefficient
            private double IN = 4; //Input layer nodes
            private double H = 5; //Hidden layer nodes
            private double Out = 3;// Output layer nodes
            private double ts = 0.001;
            private double du_1;
            private double u_1;
            private double u_2;
            private double u_3;
            private double u_4;
            private double u_5;
            private double y_1;
            private double y_2;
            private double y_3;
            public double error_1;
            public double error_2;

            private Matrix<double> wi;
            private Matrix<double> wo;
            private Matrix<double> wi_1;
            private Matrix<double> wi_2;
            private Matrix<double> wi_3;
            private Matrix<double> wo_1;
            private Matrix<double> wo_2;
            private Matrix<double> wo_3;
            private Matrix<double> xi;
            private Matrix<double> epid;
            private Matrix<double> I;
            private Matrix<double> Ohm;
            private Matrix<double> M;
            private Matrix<double> Kpid;
            private Matrix<double> dum;
            private Matrix<double> d_wo;
            private Matrix<double> segma;
            private Matrix<double> delta3m;
            private Matrix<double> delta2m;
            private Matrix<double> d_wi;

            private double[] x = new double[3];
            private double[] Oh = new double[5];
            private double kp = 0;
            private double ki = 0;
            private double kd = 0;
            private double du = 0;
            private double u = 0;
            private double dyu = 0;
            private double[] dK = new double[3];
            private double[] delta3 = new double[3];
            private double[] dO = new double[5];
            private double[] delta2 = new double[5];

            public double[] BackProp(double rin, double yout, double error)
            {
                xi = CreateMatrix.DenseOfArray(new double[,] {
                     { rin, yout, error, 1}});
                x[0] = error - error_1;
                x[1] = error;
                x[2] = error - 2 * error_1 + error_2;
                epid = CreateMatrix.DenseOfArray(new double[,] {
                       {x[0]},
                       {x[1]},
                       {x[2]}});

                I = xi * wi.Transpose();

                for (int j = 0; j < 5; j++)
                {
                    Oh[j] = (Math.Exp(I[0, j]) - Math.Exp(-I[0, j])) / (Math.Exp(I[0, j]) + Math.Exp(-I[0, j]));
                }
                Ohm = CreateMatrix.DenseOfArray(new double[,] {
                     { Oh[0], Oh[1], Oh[2], Oh[3], Oh[4]}});

                M = wo * Ohm.Transpose();

                for (int l = 0; l < 3; l++)
                {
                    M[l, 0] = Math.Exp(M[l, 0]) / (Math.Exp(M[l, 0]) + Math.Exp(-M[l, 0]));
                }

                kp = M[0, 0];
                ki = M[1, 0];
                kd = M[2, 0];


                Kpid = CreateMatrix.DenseOfArray(new double[,] {
                     { kp, ki, kd}});

                dum = Kpid * epid;
                du = dum[0, 0];
                u = u_1 + du;

                dyu = Math.Sign((yout - y_1) / du - du_1 + 0.0001);

                for (int j = 0; j < 3; j++)
                {
                    dK[j] = 2 / ((Math.Exp(M[j, 0]) + Math.Exp(-M[j, 0])) * (Math.Exp(M[j, 0]) + Math.Exp(-M[j, 0])));
                }

                for (int l = 0; l < 3; l++)
                {
                    delta3[l] = error * dyu * epid[l, 0] * dK[l];
                }

                for (int l = 0; l < 3; l++)
                {
                    for (int i = 0; i < 5; i++)
                    {
                        d_wo = xite * delta3[l] * Oh[i] + alp * (wo_1 - wo_2);
                    }
                }

                wo = wo_1 + d_wo + alp * (wo_1 - wo_2);

                for (int i = 0; i < 5; i++)
                {
                    dO[i] = 4 / ((Math.Exp(I[0, i]) + Math.Exp(-I[0, i])) * (Math.Exp(I[0, i]) + Math.Exp(-I[0, i])));
                }

                delta3m = CreateMatrix.DenseOfArray(new double[,] {
                     { delta3[0], delta3[1], delta3[2]}});
                segma = delta3m * wo;

                for (int i = 0; i < 5; i++)
                {
                    delta2[i] = dO[i] * segma[0, i];
                }

                delta2m = CreateMatrix.DenseOfArray(new double[,] {
                     { delta2[0], delta2[1], delta2[2], delta2[3], delta2[4]}});

                d_wi = xite * delta2m.Transpose() * xi;

                wi = wi_1 + d_wi + alp * (wi_1 - wi_2);

                //Update parameters
                du_1 = du;
                u_5 = u_4; u_4 = u_3; u_3 = u_2; u_2 = u_1; u_1 = u;
                y_2 = y_1; y_1 = yout;
                wo_3 = wo_2; wo_2 = wo_1; wo_1 = wo;
                wi_3 = wi_2; wi_2 = wi_1; wi_1 = wi;
                error_2 = error_1;
                error_1 = error;

                return new double[] { kp, ki, kd, u };
            }
        }

        #endregion

        #region Constructor

        public LinearPathTrackingBPNNPID()
        {
            // Initialize memory buffer
            errorQueue = new Queue<Vector<double>>();

            // Initialize BP modules
            BpX = new BPNNPID();
            BpY = new BPNNPID();
            BpZ = new BPNNPID();
        }

        #endregion

        #region Actions

        public void Init()
        {

        }

        public void Reset()
        {

        }

        public Vector<double> Control(Vector<double> xd, Vector<double> x)
        {
            double[] u = new double[3];
            var error = xd - x;
            errorQueue.Enqueue(error);

            if (errorQueue.Count >= 3)
            {
                var errors = errorQueue.ToArray();
                var n = errors.Length;
                dError = errors[n - 1] - errors[n - 2];
                dErrorInt = errors[n - 1];
                dErrorDot = errors[n - 1] - 2 * errors[n - 2] + errors[n - 3];

                BpX.error_1 = errors[n - 2][0];
                BpX.error_2 = errors[n - 3][0];
                BpY.error_1 = errors[n - 2][1];
                BpY.error_2 = errors[n - 3][1];
                BpZ.error_1 = errors[n - 2][2];
                BpZ.error_2 = errors[n - 3][2];

                var kX = BpX.BackProp(NormalizeInput(xd[0], minReference[0], maxReference[0]),
                    NormalizeInput(x[0], minReference[0], maxReference[0]), 
                    NormalizeInput(dError[0], minError[0], maxError[0]));
                var kY = BpY.BackProp(NormalizeInput(xd[1], minReference[1], maxReference[1]),
                    NormalizeInput(x[1], minReference[1], maxReference[1]),
                    NormalizeInput(dError[1], minError[1], maxError[1]));
                var kZ = BpZ.BackProp(NormalizeInput(xd[2], minReference[2], maxReference[2]),
                    NormalizeInput(x[2], minReference[2], maxReference[2]),
                    NormalizeInput(dError[2], minError[2], maxError[2]));

                //var Kp = CreateMatrix.Dense<double>(3, 3); Kp[0, 0] = kX[0]; Kp[1, 1] = kY[0]; Kp[2, 2] = kZ[0];
                //var Kd = CreateMatrix.Dense<double>(3, 3); Kd[0, 0] = kX[1]; Kd[1, 1] = kY[1]; Kd[2, 2] = kZ[1];
                //var Ki = CreateMatrix.Dense<double>(3, 3); Ki[0, 0] = kX[2]; Ki[1, 1] = kY[2]; Ki[2, 2] = kZ[2];

                u = new double[3] { kX[3], kY[3], kZ[3] };

                errorQueue.Dequeue();
            }

            return CreateVector.DenseOfArray<double>(new double[6] { u[0], u[1], u[2], 0, 0, 0 });
        }

        #endregion

        public double NormalizeInput(double input, double min, double max)
        {

            return  2 / (max - min) * (input - min) - 1;

        }


    }
}
