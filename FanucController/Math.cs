﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{
    public static class MathLib
    {
        // ---------------------------- Kinematics

        public static Matrix<double> RotationMatrix(double alpha, double beta, double gamma)
        {
            var R = CreateMatrix.Dense<double>(3,3);
            R[0, 0] = Math.Cos(alpha) * Math.Cos(beta);
            R[0, 1] = Math.Cos(alpha) * Math.Sin(beta) * Math.Sin(gamma) - Math.Sin(alpha) * Math.Cos(gamma);
            R[0, 2] = Math.Cos(alpha) * Math.Sin(beta) * Math.Cos(gamma) + Math.Sin(alpha) * Math.Sin(gamma);
            R[1, 0] = Math.Sin(alpha) * Math.Cos(beta);
            R[1, 1] = Math.Sin(alpha) * Math.Sin(beta) * Math.Sin(gamma) + Math.Cos(alpha) * Math.Cos(gamma);
            R[1, 2] = Math.Sin(alpha) * Math.Sin(beta) * Math.Cos(gamma) - Math.Cos(alpha) * Math.Sin(gamma);
            R[2, 0] = -Math.Sin(beta);
            R[2, 1] = Math.Cos(beta) * Math.Sin(gamma);
            R[2, 2] = Math.Cos(beta) * Math.Cos(gamma);
            return R;
        }

        public static Matrix<double> RotationMatrix(Vector<double> x, Vector<double> y, Vector<double> z)
        {
            var R = CreateMatrix.Dense<double>(3, 3);
            R = ((x / x.L2Norm()).ToColumnMatrix().Append((y / y.L2Norm()).ToColumnMatrix())).Append((z / z.L2Norm()).ToColumnMatrix());
            return R;
        }

        public static double[] FixedAnglesIkine(Matrix<double> R)
        {
            double[] angles = new double[3];
            angles[0] = Math.Atan2(-R[2, 0], Math.Sqrt(Math.Pow(R[0, 0], 2) + Math.Pow(R[1, 0] , 2)));
            angles[1] = Math.Atan2(R[1, 0] / Math.Cos(angles[0]), R[0, 0] / Math.Cos(angles[0]));
            angles[2] = Math.Atan2(R[2, 1] / Math.Cos(angles[0]), R[2, 2] / Math.Cos(angles[0]));
            return angles;
        }

        // ---------------------------- Linear Algebra

        public static Vector<double> PointToLinePoint(Vector<double> A, Vector<double> B, Vector<double> P)
        {
            var normal = (B - A) / (B - A).L2Norm();
            return A + ((P - A).DotProduct(normal)) * normal;

        }

        public static double PointToLineDistance(Vector<double> A, Vector<double> B, Vector<double> P)
        {
            var normal = (B - A) / (B - A).L2Norm();
            return ((P-A) - ((P-A).DotProduct(normal) * normal)).L2Norm();
        }

        public static Vector<double> CrossProduct(Vector<double> left, Vector<double> right)
        {
            var res = CreateVector.Dense<double>(3);
            res[0] = left[1] * right[2] - left[2] * right[1];
            res[1] = -left[0] * right[2] + left[2] * right[0];
            res[3] = left[0] * right[1] - left[1] * right[0];
            return res;
        }
    }

    // ---------------------------- LookUpTable

    public class LookUpTable
    {
        private int outDim;

        public LookUpTable(double[] samplePoints, double[] tableValues)
        {
            this.outDim = outDim;
        }

        public double Interpol1D(double[] points, double[] values, double qPoint)
        {
            var newPoints = points.Select(x => Math.Abs(x - qPoint)).ToArray();
            int index1 = Array.IndexOf(newPoints, newPoints.Min());
            int index2;
            double qValue = values[index1];

            if (index1 == 0 || index1 == values.Length)
            {
                return qValue;
            }

            if (points[index1] >= qPoint) { index2 = index1 - 1; }
            else
            {
                index2 = index1;
                index1 += 1;
            }

            if (index1 > 0 && index1 < values.Length)
            {
                qValue = (values[index1] - values[index2]) / (points[index1] - points[index2]) * 
                    (qPoint - points[index2]) + values[index2];
            }

            return qValue;
        }
    }
}