using System;
using System.Collections;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;

// Example 03. Modeling a layer of neurons 
// 
// A layer of neurons all receive the same input, but have different weights for each.
//     Each neuron in the layer receives all components of the input vector x.
// For each layer of neurons, we can represent the its weights as a matrix, w[row][i] 
//    where w[row] is the set of weights for the neuron in a given row.
// The scalar output of each neuron is computed as in Example 02, as the non-linear response to the 
//    weighted sum of the inputs.
// The output of the layer is now a vector, y[row]
//
// We can compute the output of the entire layer of neurons by applying the activation function
//    to each element of the product of matrix W and vector x.
//    

namespace NNDotNetTutorial
{
    class Example03
    {
        static void Main(string[] args)
        {
            // x is a vector of 10 doubles, selected randomly from [0,1]
            var x = Vector.Build.Dense(Generate.Uniform(10));

            #region printstuff
            Console.WriteLine("Vector x[{0}] = ", x.Count);
            foreach (var xi in x) { Console.Write("{0:N2} ", xi); }
            Console.WriteLine("\n"); 
            #endregion


            #region InitRandomGenerator
            var randomSeq = MersenneTwister.DoubleSequence(42);
            var randomEnumerator = randomSeq.GetEnumerator();
            randomEnumerator.MoveNext();
            
            #endregion

            // w is a 10x10 matrix of 10 doubles, selected randomly from [0,1]
            var W = Matrix.Build.Dense(10, 10);

            #region InitializeWithRandomValues
            for (int row = 0; row < 10; row++)
            {
                for (int col = 0; col < 10; col++)
                {
                    W[row, col] = randomEnumerator.Current;
                    randomEnumerator.MoveNext();

                }
            } 
            #endregion

            #region printstuff
            Console.WriteLine("Matrix W[{0},{1}] = ", W.RowCount, W.ColumnCount);
            foreach (var row in W.EnumerateRows())
            {
                foreach (var ci in row) { Console.Write("{0:N2} ", ci); }
                Console.WriteLine();
            }
            Console.WriteLine("\n"); 
            #endregion

            var y = W * x; // vector result of weighted sum for each neuron

            #region printstuff
            Console.WriteLine("Vector (W*x)[{0}] = ", y.Count);
            foreach (var yi in y) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n"); 
            #endregion

            y.MapInplace(SpecialFunctions.Logistic);   // apply the activation function
            
            #region printstuff
            Console.WriteLine("Vector y[{0}] = ", y.Count);
            foreach (var yi in y) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");

            #endregion
            
            return;
        }
    }
}

