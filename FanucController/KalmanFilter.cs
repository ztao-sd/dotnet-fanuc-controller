using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{

    public class KalmanFilter
    {
        protected double CtrackInterval = 1 / 29;

        // Prediction
        protected Matrix<double> A; // State transition matrix
        protected Matrix<double> predP; // Predicted last error covariance matrix
        protected Matrix<double> P_; // Last error covariance matrix
        protected Matrix<double> Q; // Process noise covariance matrix
        protected Vector<double> predX; // Predicted state
        protected Vector<double> X_; // Predict last state

        // Estimation
        protected Matrix<double> K; // Kalman gain
        protected Matrix<double> R; // Measurement noise covariance
        protected Matrix<double> P; // Error covariance matrix
        protected Matrix<double> I; // Identity matrix
        protected Vector<double> X; // Estimated state
        protected Vector<double> Z; // Measurement vector

        protected Matrix<double> S; // Intermediate matrix

        // Measurement
        protected Matrix<double> H; // Observation matrix

        public KalmanFilter()
        {
            // State transition matrix
            A = CreateMatrix.DenseIdentity<double>(12);
            for (int i = 0; i < 6; i++)
            {
                A[i, 6 + i] = CtrackInterval;
            }
            // Measurment noise matrix
            R = CreateMatrix.DenseDiagonal<double>(6, 0.001);
            double[] r = new double[6] { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
            for (int i = 0; i < 6; i++)
            {
                R[i, i] = r[i];
            }
            P_ = CreateMatrix.DenseDiagonal<double>(12, 0.05);
            Q = CreateMatrix.DenseDiagonal<double>(12, 0.00001);
            I = CreateMatrix.DenseIdentity<double>(12);
            H = CreateMatrix.DenseIdentity<double>(6, 12);
        }

        public virtual void Initialize(Vector<double> initPose)
        {
            // Initial state
            X_ = CreateVector.Dense<double>(12);
            initPose.CopySubVectorTo(X_, 0, 0, 6);
            // Clear variables
            X = CreateVector.Dense<double>(12);
        }

        public virtual Vector<double> Estimate(Vector<double> pose)
        {
            // Measurement
            Z = pose;
            // Prediction
            predX = A * X_;
            predP = A * P_ * A.Transpose() + Q;
            // Estimation
            S = (H * predP * H.Transpose() + R).Inverse();
            K = predP * H.Transpose() * S;
            X = predX + K * (Z - H * predX);
            // Update last variables
            P_ = (I - K * H) * predP;
            X_ = X;

            return X;
        }
    }

    public class RobustKalmanFilter : KalmanFilter
    {
        private double alpha_;
        private double alpha;
        private double beta;
        private double omega;
        private Matrix<double> barP;
        private Vector<double> d_;
        private Vector<double> d;
        //private Matrix<double> rP;

        public RobustKalmanFilter() : base()
        {
            alpha_ = 1.0;
            beta = 0.99;
            omega = 0.99;
            d_ = CreateVector.Dense<double>(6);
        }

        public override void Initialize(Vector<double> initPose)
        {
            base.Initialize(initPose);
        }

        public override Vector<double> Estimate(Vector<double> pose)
        {
            // Measurement
            Z = pose;
            // Prediction
            predX = A * X_;
            d = Z - H * predX;
            barP = omega * (d_.ToColumnMatrix() * d_.ToRowMatrix()) + (1 - omega) * (d.ToColumnMatrix() * d_.ToRowMatrix());
            predP = A * P_ * A.Transpose() + Q;
            if (barP.Trace() < predP.Trace())
            {
                alpha = 1;
            }
            else
            {
                alpha = predP.Trace() / barP.Trace();
            }
            alpha = beta * alpha_ + (1 - beta) * alpha;
            predP = predP / alpha;
            // Estimation
            S = (H * predP * H.Transpose() + R).Inverse();
            K = predP * H.Transpose() * S;
            X = predX + K * (Z - H * predX);
            // Update last variables
            P_ = (I - K * H) * predP / alpha;
            X_ = X;
            d_ = d;
            alpha_ = alpha;

            return X;
        }

    }

}
