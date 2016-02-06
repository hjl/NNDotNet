using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;


// Example 01. Vector and Matrices using MathNet.Numerics
// 
// We will use the MathNet.Numerics library (http://numerics.mathdotnet.com) to provide linear algebra functions.
// Use the NuGet package manager to add it to your project if it is missing.

namespace NNDotNetTutorial
{
    class Example01
    {
        static void Main(string[] args)
        {
            // Example of constructing vectors and matrices. In this case, 
            //     x is a vector of 10 doubles, all with value 0.0
            //     w is a vector of 10 doubles, selected randomly from [0,1]
            //     A is an identity matrix of 10 x 10 floats
            var x = Vector.Build.Dense(10, 0.0);
            var w = Vector.Build.Dense(Generate.Uniform(10));

            var A = Matrix.Build.DenseIdentity(10);

            // Vectors and matrix arithmetic is extended in the expected way, so you can just write "+", "-", "*"
            // 
            var y = x * A;

            #region printstuff
            Console.WriteLine("Vector x[{0}] = ", x.Count);
            foreach (var xi in x) { Console.Write("{0:N2} ", xi); }
            // can use x.ToString() instead, but that will output as vertically formatted column vector.
            Console.WriteLine("\n");

            Console.WriteLine("Vector w[{0}] = ", w.Count);
            foreach (var wi in w) { Console.Write("{0:N2} ", wi); }
            Console.WriteLine("\n");

            Console.WriteLine("Matrix A[{0},{1}] = ", A.RowCount, A.ColumnCount);
            foreach (var row in A.EnumerateRows())
            {
                foreach (var ci in row) { Console.Write("{0:N2} ", ci); }
                Console.WriteLine();
            }
            Console.WriteLine("\n");

            Console.WriteLine("Vector y[{0}] = ", y.Count);
            foreach (var yi in w) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");
            #endregion

            return;
        }
    }
}