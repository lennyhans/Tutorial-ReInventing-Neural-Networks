using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralXOR
    {
        public static void Main(string[] args)
        {
            // Current iteration process
            int Iteration = 0;


            NeuralNetwork BestNetwork = new NeuralNetwork( 
                new uint[] { 2, 2, 1 }); // The best network currently made
            double BestCost = double.MaxValue; // The cost that the best network achieve
            double[] BestNetworkResults = new double[4]; // The results that the best network calculated

            double[][] Inputs = new double[][]
            // This represents the possible inputs or the training dataset
            {
                new double[] {0,0 },
                new double[] {1,0 },
                new double[] {0,1 },
                new double[] {1,1 }
            };
            //This represents the expected outputs from the optimum NeuralNetwork
            double[] ExpectedOutputs = new double[] { 0, 1, 1, 0 };

            // while vars
            double LimitSearchCost = 0.002d;
            for (; BestCost >= LimitSearchCost; Iteration++ )
                //Training forever!
            {
                //Clone the current best network
                NeuralNetwork MutatedNetwork = new NeuralNetwork(BestNetwork);

                MutatedNetwork.Mutate();
                double MutatedNetworkCost = 0;
                // The result that the mutated network calculated
                double[] CurrentNetworkResult = new double[4];

                //Calculate the cost of the mutated network
                for(int i = 0; i< Inputs.Length; i++)
                {
                    double[] Result = MutatedNetwork.FeedForward(Inputs[i]);
                    MutatedNetworkCost += Math.Abs(Result[0] - ExpectedOutputs[i]);

                    CurrentNetworkResult[i] = Result[0];
                }

                //Does the mutated network perform better than the last one ?
                if(MutatedNetworkCost < BestCost)
                {
                    BestNetwork = MutatedNetwork;
                    BestCost = MutatedNetworkCost;
                    BestNetworkResults = CurrentNetworkResult;
                }

                // Print only each 20.000 iteration in order to speed up the training process
                if(Iteration % 20000 == 0 || LimitSearchCost >= BestCost)
                {
                    Console.Clear();

                    for(int i = 0;  i < BestNetworkResults.Length; i++)
                    {
                        Console.WriteLine(string.Format("{0},{1} | {2}",
                            Inputs[i][0],
                            Inputs[i][1],
                            BestNetworkResults[i].ToString("N17")));
                    }
                    Console.WriteLine(string.Format("Cost : {0}",
                        BestCost));
                    Console.WriteLine(string.Format("Iteration : {0}",
                        Iteration
                        ));
                }
            }
        }
    }
}
