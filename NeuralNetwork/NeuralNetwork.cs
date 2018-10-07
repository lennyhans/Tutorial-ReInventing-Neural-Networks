using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public UInt32[] Topology
        {
            get
            {
                UInt32[] Result = new uint[TheTopology.Count];
                TheTopology.CopyTo(Result, 0);
                return Result;
            }
        }
        ReadOnlyCollection<UInt32> TheTopology;
        NeuralSection[] Sections;
        Random TheRandomizer;

        /// <summary>
        /// Initiates a NeuralNetwork from a Topology and a Seed.
        /// </summary>
        /// <param name="Topology">The Topology of the Neural Network.</param>
        /// <param name="Seed">The Seed of the Neural Network.
        /// Set to 'null' to use a Timed Seed.</param>
        public NeuralNetwork(UInt32[] Topology, Int32? Seed = 0)
        {
            // Validation check
            if (Topology.Length < 2)
                throw new ArgumentOutOfRangeException("A neural network cannot contain less than 2 layers.",
                    "Topology");
            for(int i=0; i< Topology.Length; i++)
            {
                if (Topology[i] < 1)
                    throw new ArgumentOutOfRangeException("Topology", "A single layer of neurons must contain," +
                        "at least, one neuron");

            }
            TheRandomizer = Seed.HasValue ? new Random(Seed.Value) : new Random();

            TheTopology = new List<uint>(Topology).AsReadOnly();

            Sections = new NeuralSection[TheTopology.Count - 1];

            for( int i = 0; i < Sections.Length; i++) {
                Sections[i] = new NeuralSection(TheTopology[i], TheTopology[i + 1], TheRandomizer);
            }
                
        }
        /// <summary>
        /// Initiates an independent Deep-Copy of the Neural Network provided.
        /// </summary>
        /// <param name="Main">The Neural Network that should be cloned.</param>
        public NeuralNetwork(NeuralNetwork Main)
        {
            // Init Random
            TheRandomizer = new Random(Main.TheRandomizer.Next());

            // Set Topology
            TheTopology = Main.TheTopology;

            // Initialize Sections 
            Sections = new NeuralSection[TheTopology.Count - 1];

            // Set the Sections
            for ( int i = 0; i < Sections.Length; i++)
            {
                Sections[i] = new NeuralSection(Main.Sections[i]);
            }
        }

        public double[] FeedForward(double[] Input)
        {
            // Validate Input
            if (Input == null)
                throw new ArgumentNullException("Input", "The input array cannot be set to null.");
            else if (Input.Length != TheTopology[0])
                throw new ArgumentOutOfRangeException("Input", "The input array's lenght does not match the " +
                    "number of neurons in the input layer.");
            double[] Output = Input;

            // Feed value through all sections
            for(int i = 0; i < Sections.Length; i++)
            {
                Output = Sections[i].FeedForward(Output);
            }

            return Output;
        }
        /// <summary>
        /// Mutate the NeuralNetwork
        /// </summary>
        /// <param name="MutationProbability">The probability that a weight is going to be mutated.
        /// (Range 0-1)</param>
        /// <param name="MutationAmount">The Maximum amount a mutated weight would change.</param>
        public void Mutate( double MutationProbability = 0.3, double MutationAmount = 2.0)
        {
            for (int i = 0; i < Sections.Length; i++)
                Sections[i].Mutate(MutationProbability, MutationAmount);
        }


        private class NeuralSection
        {
            /// <summary>
            /// Contains all the weights of a section where [i][j] represents the weight
            /// from neuron i in the input layer and neuron j in the output layer
            /// </summary>
            private double[][] Weights;
            /// <summary>
            /// Contains a reference to the Random instance of the NeuralNetwork
            /// </summary>
            private Random TheRandomizer; 

            /// <summary>
            /// Init a NeuralSection from a topology and seed
            /// </summary>
            /// <param name="InputCount">The number of input neurons in the section.</param>
            /// <param name="OutputCount">The number of output neurons in the section.</param>
            /// <param name="Randomizer">The random  instance of the NeuralNetwork.</param>
            public NeuralSection(UInt32 InputCount, UInt32 OutputCount, Random Randomizer)
            {
                // Validation checks
                if (InputCount == 0)
                    throw new ArgumentOutOfRangeException("InputCount", "You cannot create a Neural Layer " +
                        "with no input neurons.");
                else if (OutputCount == 0)
                    throw new ArgumentOutOfRangeException("OutputCount", "You cannot create a Neural Layer " +
                        "with no output neurons.");
                else if (Randomizer == null)
                    throw new ArgumentNullException("Randomizer", "The randomizer cannot be set to null");

                // Set randomizer
                TheRandomizer = Randomizer;

                // Init Weights
                Weights = new double[InputCount + 1][]; // +1 for the Bias Neuron

                for (int i = 0; i < Weights.Length; i++)
                    Weights[i] = new double[OutputCount];

                // Set Random Weights
                for (int i = 0, j; i < Weights.Length; i++)
                    for (j = 0; j < Weights[i].Length; j++)
                        Weights[i][j] = TheRandomizer.NextDouble() - .5f;
            }
            /// <summary>
            /// Init an independent Deep-copy of the NeuralSection provided.
            /// </summary>
            /// <param name="Main">The NeuralSection that should be cloned.</param>
            public NeuralSection(NeuralSection Main)
            {
                // Set randomizer
                TheRandomizer = Main.TheRandomizer;

                // Init Weights
                Weights = new double[Main.Weights.Length][];
                
                for (int i = 0; i < Weights.Length; i++)
                    Weights[i] = new double[Main.Weights[0].Length];

                // Set weights
                for (int i = 0, j; i < Weights.Length; i++)
                    for (j = 0; j < Weights[i].Length; j++)
                        Weights[i][j] = Main.Weights[i][j];
            }
            /// <summary>
            /// Feed input throgh the NeuralSection and get the output.
            /// </summary>
            /// <param name="input">The values to set the input neurons.</param>
            /// <returns>The values in the output neurons after propagation.</returns>
            public double[] FeedForward(double[] Input)
            {
                // Validation checks
                if (Input == null)
                    throw new ArgumentNullException("Input", "The input array cannot be set to null.");
                else if (Input.Length != Weights.Length - 1)
                    throw new ArgumentOutOfRangeException("Input", "The input array's length does not match" +
                        "the number of neurons in the input layer");
                // Init Output Array
                double[] Output = new double[Weights[0].Length];

                // Calculate Value
                for(int i = 0, j; i < Weights.Length; i++)
                    for(j = 0; j< Weights[i].Length; j++)
                    {
                        if (i == Weights.Length - 1) // If is the bias neuron
                            Output[j] += Weights[i][j]; // then the value of the neuron is 1
                        else
                            Output[j] += Weights[i][j] * Input[i];
                    }
                // Apply activation function
                for (int i = 0; i < Output.Length; i++)
                    Output[i] = ReLU(Output[i]);


                return Output;
            }

            /// <summary>
            /// Mutate the NeuralSection
            /// </summary>
            /// <param name="MutationProbability">The probability that a weight is going to be mutated.
            /// (Ranges 0-1)</param>
            /// <param name="MutationAmount">The maximum amount a Mutated Weigth would change</param>
            public void Mutate(double MutationProbability, double MutationAmount)
            {
                for (int i = 0, j; i < Weights.Length; i++)
                    for (j = 0; j < Weights[i].Length; j++)
                        if (TheRandomizer.NextDouble() < MutationProbability)
                            Weights[i][j] = TheRandomizer.NextDouble() * (MutationAmount * 2) - MutationAmount;
            }

            /// <summary>
            /// Puts a double through the activation function ReLU
            /// </summary>
            /// <param name="x">The value to put through the function</param>
            /// <returns> x after it is put through ReLU</returns>
            public double ReLU(double x)
            {
                return (x >= 0) ? x : x / 20;
            }
        }
    }
}
