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
            double r_p = 0.003;
            double r_r = 0.0006;
            //double r_p = 0.001;
            //double r_r = 0.001;
            double[] r = new double[6] { r_p, r_p, r_p, r_r, r_r, r_r };
            for (int i = 0; i < 6; i++)
            {
                R[i, i] = r[i];
            }
            P_ = CreateMatrix.DenseDiagonal<double>(12, 0.001);
            Q = CreateMatrix.DenseDiagonal<double>(12, 0.00001);
            double q_p = 0.001;
            double q_p_d = 0.001;
            double q_o = 0.0002;
            double q_o_d = 0.0001;
            //double q_p = 0.00001;
            //double q_p_d = 0.00001;
            //double q_o = 0.00001;
            //double q_o_d = 0.00001;
            double[] q = new double[12] { q_p, q_p, q_p, q_o, q_o, q_o, q_p_d, q_p_d, q_p_d, q_o_d, q_o_d, q_o_d };
            for (int i = 0; i < 12; i++)
            {
                Q[i, i] = q[i];
            }
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
            alpha_ = 0;
            //alpha_ = 0.9;
            beta = 0.995;
            omega = 0.90;
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
            barP = omega * (d_.ToColumnMatrix() * d_.ToRowMatrix()) + (1 - omega) * (d.ToColumnMatrix() * d.ToRowMatrix());
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

    public class FilterTest
    {
        public string CsvPath;
        public string KfPath;
        public string RkfPath;
        public KalmanFilter KalmanFilter;
        public KalmanFilter RobustKalmanFilter;
        public List<PoseData> PoseDataList;
        public bool KalmanFilterEnabled = false;
        public bool RobustKalmanFilterEnabled = false;

        public FilterTest(string csvPath, string kfPath, string rkfPath)
        {
            CsvPath = csvPath;
            KfPath = kfPath;
            RkfPath = rkfPath;
            KalmanFilter = new KalmanFilter();
            RobustKalmanFilter = new RobustKalmanFilter();
            PoseDataList = Csv.ReadCsv<PoseData>(CsvPath);
        }

        public Vector<double> ApplyKalmanFilter(Vector<double> newPose)
        {
            if (!KalmanFilterEnabled)
            {
                KalmanFilter.Initialize(newPose);
                KalmanFilterEnabled = true;
                return newPose;
            }
            else
            {
                Vector<double> _poseTemp = KalmanFilter.Estimate(newPose);
                return _poseTemp;
            }
        }

        public Vector<double> ApplyRobustKalmanFilter(Vector<double> newPose)
        {
            if (!RobustKalmanFilterEnabled)
            {
                RobustKalmanFilter.Initialize(newPose);
                RobustKalmanFilterEnabled = true;
                return newPose;
            }
            else
            {
                Vector<double> _poseTemp = RobustKalmanFilter.Estimate(newPose);
                return _poseTemp;
            }
        }

        public void Test()
        {
            var kfList = new List<PoseData>();
            var rkfList = new List<PoseData>();

            foreach (var poseData in PoseDataList)
            {
                var newPose = poseData.ToVector();
                var time = poseData.Time;
                var kfPose = ApplyKalmanFilter(newPose);
                var rkfPose = ApplyRobustKalmanFilter(newPose);
                kfList.Add(new PoseData(kfPose, time));
                rkfList.Add(new PoseData(rkfPose, time));
                Csv.WriteCsv(KfPath, kfList);
                Csv.WriteCsv(RkfPath, rkfList);
            }
        }

    }

}
