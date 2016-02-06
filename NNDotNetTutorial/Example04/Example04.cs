using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;


// Example 04. Two stacked layers of neurons 
// 
// A layer of neurons all receive the same input, but have different weights for each.
//     Each neuron in the layer receives all components of the input vector x.
// For each layer of neurons, we can represent the its weights as a matrix, w[row][i] 
//    where w[row] is the set of weights for the neuron in a given row.
// The scalar output of each neuron is computed as in Example 02, as the non-linear response to the 
//    weighted sum of the inputs.
// The output of the layer is now a vector, y[row]. But the output of the first layer is "hidden"
//    with only the second (last) layer being exposed as a function output.
//
// We can compute the output of the entire layer of neurons by applying the activation function
//    to each element of the product of matrix W and vector x.
//    

namespace NNDotNetTutorial
{
    class Example04
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
            var W0 = Matrix.Build.Dense(10, 10);
            
            #region InitMatrixW0WithRandom
            for (int row = 0; row < 10; row++)
            {
                for (int col = 0; col < 10; col++)
                {
                    W0[row, col] = randomEnumerator.Current;
                    randomEnumerator.MoveNext();

                }
            } 
            #endregion

            var W1 = Matrix.Build.Dense(10, 10);

            #region InitMatrixW1WithRandom
            for (int row = 0; row < 10; row++)
            {
                for (int col = 0; col < 10; col++)
                {
                    W1[row, col] = randomEnumerator.Current;
                    randomEnumerator.MoveNext();

                }
            }
            
            #endregion

            #region printstuff
            Console.WriteLine("Matrix W0[{0},{1}] = ", W0.RowCount, W0.ColumnCount);
            foreach (var row in W0.EnumerateRows())
            {
                foreach (var ci in row) { Console.Write("{0:N2} ", ci); }
                Console.WriteLine();
            }
            Console.WriteLine("\n");

            #endregion

            var y0 = W0 * x; // vector result of weighted sum for each neuron

            #region printstuff
            Console.WriteLine("Vector (W0*x)[{0}] = ", y0.Count);
            foreach (var yi in y0) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");

            #endregion

            y0.MapInplace(SpecialFunctions.Logistic);   // apply the activation function
           
            #region printstuff
            Console.WriteLine("Vector y0[{0}] = ", y0.Count);
            foreach (var yi in y0) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");

            Console.WriteLine("Matrix W1[{0},{1}] = ", W1.RowCount, W1.ColumnCount);
            foreach (var row in W1.EnumerateRows())
            {
                foreach (var ci in row) { Console.Write("{0:N2} ", ci); }
                Console.WriteLine();
            }
            Console.WriteLine("\n");

            #endregion

            var y1 = W1 * y0; // use previous layer output as input to the next layer

            #region printstuff
            Console.WriteLine("Vector (W1*y0)[{0}] = ", y1.Count);
            foreach (var yi in y1) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");
            #endregion

            y1.MapInplace(SpecialFunctions.Logistic);   // apply the activation function

            #region printstuff
            Console.WriteLine("Vector y1[{0}] = ", y1.Count);
            foreach (var yi in y1) { Console.Write("{0:N2} ", yi); }
            Console.WriteLine("\n");
            #endregion

            return;
        }
    }
}

