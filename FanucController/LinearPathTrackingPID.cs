using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace FanucController
{
    public class LinearPathTrackingPID
    {
        public int Dim;

        // Gains
        public Matrix<double> Kp;
        public Matrix<double> Ki;
        public Matrix<double> Kd;

        // Error List
        public List<Vector<double>> Errors;

        // Control
        public Vector<double> ControlSignal;

        #region Constructor

        public LinearPathTrackingPID()
        {
            Dim = 3;
            Kp = CreateMatrix.DenseOfDiagonalArray(new double[3] {0.2, 0.2, 0.1});
            Ki = CreateMatrix.DenseDiagonal(Dim, 0.005);
            Kd = CreateMatrix.DenseDiagonal(Dim, 0.05);
            Errors = new List<Vector<double>>();
            ControlSignal = CreateVector.Dense<double>(Dim);
        }

        #endregion

        #region Actions

        public void Init()
        {
            Errors.Clear();
            Errors.Add(CreateVector.Dense<double>(Dim));
            Errors.Add(CreateVector.Dense<double>(Dim));
            Errors.Add(CreateVector.Dense<double>(Dim));
            ControlSignal.Clear();
        }

        public void Reset()
        {
            Errors.Clear();
            Errors.Add(CreateVector.Dense<double>(Dim));
            Errors.Add(CreateVector.Dense<double>(Dim));
            Errors.Add(CreateVector.Dense<double>(Dim));
            ControlSignal.Clear();
        }

        public Vector<double> Control(Vector<double> error)
        {
            // Add & delete from error list
            Errors.Insert(0, error);
            Errors.RemoveAt(Errors.Count - 1);

            ControlSignal += Kp * (Errors[0] - Errors[1]) + Ki * (Errors[0]) + Kd * (Errors[0] - 2 * Errors[1] + Errors[2]);
            Vector<double> control = CreateVector.Dense<double>(6);
            ControlSignal.CopySubVectorTo(control, 0, 0, 3);

            return control;
        }

        #endregion

    }
}
