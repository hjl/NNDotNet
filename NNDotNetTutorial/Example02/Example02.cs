using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;

// Example 02. Modeling a single neuron 
// 
// The inputs to our model neuron are the components of vector x
// Each input has a separate linear weight w, constributing to the sum(w[i]x[i]) 
// The weighted sum is used as the input to a non-linear activation function.
// In this example we will use the sigmoid logistic function of the weighted sum to compute the activation y.
// Note that y is a scalar.

namespace NNDotNetTutorial
{
    class Example02
    {
        static void Main(string[] args)
        {
            // x is a vector of 10 doubles, selected randomly from [0,1]
            // w is a vector of 10 doubles, selected randomly from [0,1]
            var w = Vector.Build.Dense(Generate.Uniform(10));
            var x = Vector.Build.Dense(Generate.Uniform(10));

            var wdotx = w * x;
            var y = SpecialFunctions.Logistic(wdotx);

            #region printstuff
            Console.WriteLine("Vector x[{0}] = ", x.Count);
            foreach (var xi in x) { Console.Write("{0:N2} ", xi); }
            Console.WriteLine("\n");

            Console.WriteLine("Vector w[{0}] = ", w.Count);
            foreach (var wi in w) { Console.Write("{0:N2} ", wi); }
            Console.WriteLine("\n");

            Console.WriteLine("Scalar w * x = {0:N2}, y = sigmoid(w * x) = {1:N2}", wdotx, y);
            
            #endregion

            return;
        }
    }
}
