using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{


    public class BPNNPID
    {

        public double kLimits;

        // Hyperparameters of NN
        public double xite = 0.08; // Learning rate
        public double alp = 0.04; // Inertia coefficient
        public double IN = 4; // Input layer nodes
        public double H = 5; // Hidden layer nodes
        public double Out = 3;// Output layer nodes

        // Control
        public double du_1; // Delta u (control)
        public double u_1; // u @ t-1
        //public double u_2; // u @ t-2
        //public double u_3; // u @ t-3
        //public double u_4; // u @ t-4
        //public double u_5; // u @ t-5

        // System output
        public double y_1; // y @ t-1
        //public double y_2; // y @ t-2
        //public double y_3; // y @ t-3
        public double error_1;
        public double error_2;

        // public double ts = 0.001; // Time interval

        // Neural network parameters
        public Matrix<double> wi; // Hidden layer parameters
        public Matrix<double> wo; // Output layer parameters
        public Matrix<double> wi_1; // wi @ t-1
        public Matrix<double> wi_2; // wi @ t-2
        // public Matrix<double> wi_3; // wi @ t-3
        public Matrix<double> wo_1; // wo @ t-1
        public Matrix<double> wo_2; // wo @ t-2
        // public Matrix<double> wo_3; // wo @ t-3

        // Intermediate variables
        public Matrix<double> xi; // Neural network input
        public Matrix<double> epid; // e, edot, int_e
        public Matrix<double> I; // xi*wi'
        public Matrix<double> Ohm; // 
        public Matrix<double> M;
        public Matrix<double> Kpid;
        public Matrix<double> dum;
        public Matrix<double> d_wo;
        public Matrix<double> segma;
        public Matrix<double> delta3m;
        public Matrix<double> delta2m;
        public Matrix<double> d_wi;


        public double[] x = new double[3];
        public double[] Oh = new double[5];
        public double kp = 0;
        public double ki = 0;
        public double kd = 0;
        public double du = 0;
        public double u = 0;
        public double dyu = 0;
        public double[] dK = new double[3];
        public double[] delta3 = new double[3];
        public double[] dO = new double[5];
        public double[] delta2 = new double[5];

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
            // wi_3 = wi;

            // Output weights
            wo = CreateMatrix.DenseOfArray(new double[,] {
                { 0.0438, 0.1314, 0.1183, 0.4981,0.0008},
                { 0.4795,-0.2146, 0.1486,-0.1219,0.4572},
                {-0.3443, 0.4441, 0.2451,-0.4927,-0.3680}});
            wo_1 = wo;
            wo_2 = wo;
            // wo_3 = wo;

            // Errors
            error_1 = 0;
            error_2 = 0;

            // Control
            u = 0;
            u_1 = 0;
            du_1 = 0;
            kLimits = 0.05;

        }

        public void Reset()
        {
            // Input weights
            wi = CreateMatrix.DenseOfArray(new double[,] {
                { 0.3147,-0.4025,-0.3424,-0.3581},
                { 0.4058,-0.2215, 0.4706,-0.0782},
                {-0.3730, 0.0469, 0.4572, 0.4157},
                { 0.4134, 0.4575,-0.0146, 0.2922},
                { 0.1324, 0.4649, 0.3003, 0.4595}});
            wi_1 = wi;
            wi_2 = wi;
            // wi_3 = wi;

            // Output weights
            wo = CreateMatrix.DenseOfArray(new double[,] {
                { 0.0438, 0.1314, 0.1183, 0.4981,0.0008},
                { 0.4795,-0.2146, 0.1486,-0.1219,0.4572},
                {-0.3443, 0.4441, 0.2451,-0.4927,-0.3680}});
            wo_1 = wo;
            wo_2 = wo;
            // wo_3 = wo;

            // Errors
            error_1 = 0;
            error_2 = 0;

            // Control
            u = 0;
            u_1 = 0;
            du_1 = 0;

            // Pid coefficients
            kp = 0;
            ki = 0;
            kd = 0;
            du = 0;
            u = 0;
            dyu = 0;

            // Variables
        }
        
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

            kp = M[0, 0] * kLimits;
            ki = M[1, 0] * kLimits;
            kd = M[2, 0] * kLimits;


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
            // u_5 = u_4; u_4 = u_3; u_3 = u_2; u_2 = u_1; 
            u_1 = u;
            // y_2 = y_1; 
            y_1 = yout;
            //wo_3 = wo_2; 
            wo_2 = wo_1; wo_1 = wo;
            // wi_3 = wi_2; 
            wi_2 = wi_1; wi_1 = wi;
            error_2 = error_1;
            error_1 = error;

            return new double[] { kp, ki, kd, u };
        }
    }


    public class PathTrackingBPNNPID
    {
        // Memory Buffer
        public Queue<Vector<double>> errorQueue;

        // Back Propagation
        public BPNNPID[] Bp;
        public BPNNPID BpX;
        public BPNNPID BpY;
        public BPNNPID BpZ;

        // Limits
        public double[] minReference = new double[3] { -3000, -3000, -3000 };
        public double[] maxReference = new double[3] { 3000, 3000, 3000 };
        public double[] minError = new double[3] { -0.20, -0.20, -0.20 };
        public double[] maxError = new double[3] { 0.20, 0.20, 0.20 };

        // PID Constants
        public double[] kValues;

        #region Constructor

        public PathTrackingBPNNPID()
        {
            // Initialize memory buffer
            // errorQueue = new Queue<Vector<double>>();

            // Initialize BP modules
            Bp = new BPNNPID[3];

            double[] k = new double[3] { 0.01, 0.01, 0.001 };

            for (int i = 0; i < 3; i++)
            {
                Bp[i] = new BPNNPID();
                Bp[i].kLimits = k[i];
            }

            kValues = new double[9];
        }

        #endregion

        #region Actions

        public void Init()
        {

        }

        public void Reset()
        {
            for (int i = 0; i < 3; i++)
            {
                Bp[i].Reset();
            }
        }

        public Vector<double> Control(Vector<double> reference, Vector<double> actual, Vector<double> error)
        {
            Vector<double> control = CreateVector.Dense<double>(6);

            double xd; double x; double e;
            double[] u = new double[3];
            double[] kp = new double[3]; 
            double[] ki = new double[3]; 
            double[] kd = new double[3];
            for (int i = 0; i < 3; i++)
            {
                xd = NormalizeInput(reference[i], minReference[i], maxReference[i]);
                x = NormalizeInput(actual[i], minReference[i], maxReference[i]);
                e = error[i];
                var result = Bp[i].BackProp(xd, x, e);
                u[i] = result.Last();
                kp[i] = result[0];
                ki[i] = result[1];
                kd[i] = result[2];
            }
            var temp = CreateVector.DenseOfArray(u);
            temp.CopySubVectorTo(control, 0, 0, 3);

            // Assign kValues
            for (int i = 0; i < 3; i++)
            {
                kValues[i] = kp[i];
                kValues[i + 3] = ki[i];
                kValues[i + 6] = kd[i];
            }

            return control;
        }

        #endregion

        public double NormalizeInput(double input, double min, double max)
        {

            return  2 / (max - min) * (input - min) - 1;

        }

        public double NormalizeOutput(double output, double min, double max)
        {

            return (max - min) / 2 * (output + 1) + min;

        }


    }

    public class PathTrackingBPNNPID6D
    {
        // Memory Buffer
        public Queue<Vector<double>> errorQueue;

        // Back Propagation
        public BPNNPID[] Bp;
        public BPNNPID BpX;
        public BPNNPID BpY;
        public BPNNPID BpZ;
        public BPNNPID BpW;
        public BPNNPID BpP;
        public BPNNPID BpR;

        // Limits
        public double[] minReference = new double[6] { -1800, -1200, -100, 2.90, -0.15, 1.4 };
        public double[] maxReference = new double[6] { -1600, -400, 0, 3.20, 0.30, 1.60};
        public double[] minError = new double[6] { -0.50, -0.50, -0.50, -0.002, -0.002, -0.002 };
        public double[] maxError = new double[6] { 0.50, 0.50, 0.50, 0.002, 0.002, 0.002 };

        // PID Constants
        public double[] kValues;

        #region Constructor

        public PathTrackingBPNNPID6D()
        {
            // Initialize memory buffer
            // errorQueue = new Queue<Vector<double>>();

            // Initialize BP modules
            Bp = new BPNNPID[6];

            double[] k = new double[6] { 0.01, 0.01, 0.001, 57.3 * 0.01, 57.3 * 0.001, 57.3 * 0.01 };

            for (int i = 0; i < 6; i++)
            {
                Bp[i] = new BPNNPID();
                Bp[i].kLimits = k[i];
            }

            kValues = new double[18];
        }

        #endregion

        #region Actions

        public void Init()
        {

        }

        public void Reset()
        {
            for (int i = 0; i < 6; i++)
            {
                Bp[i].Reset();
            }
        }

        public Vector<double> Control(Vector<double> reference, Vector<double> actual, Vector<double> error)
        {
            Vector<double> control = CreateVector.Dense<double>(6);

            double xd; double x; double e;
            double[] u = new double[6];
            double[] kp = new double[6];
            double[] ki = new double[6];
            double[] kd = new double[6];
            for (int i = 0; i < 6; i++)
            {
                xd = NormalizeInput(reference[i], minReference[i], maxReference[i]);
                x = NormalizeInput(actual[i], minReference[i], maxReference[i]);
                e = error[i];
                var result = Bp[i].BackProp(xd, x, e);
                u[i] = result.Last();
                kp[i] = result[0];
                ki[i] = result[1];
                kd[i] = result[2];
            }
            var temp = CreateVector.DenseOfArray(u);
            temp.CopySubVectorTo(control, 0, 0, 6);

            // Assign kValues
            for (int i = 0; i < 6; i++)
            {
                kValues[i] = kp[i];
                kValues[i + 6] = ki[i];
                kValues[i + 12] = kd[i];
            }

            return control;
        }

        #endregion

        public double NormalizeInput(double input, double min, double max)
        {

            return 2 / (max - min) * (input - min) - 1;

        }

        public double NormalizeOutput(double output, double min, double max)
        {

            return (max - min) / 2 * (output + 1) + min;

        }


    }
}
