using System;

namespace NeuralXORConsole
{
    class NeuralXORConsole
    {
        static void Main(string[] args)
        {
            //https://www.codeproject.com/Articles/1220276/ReInventing-Neural-Networks
            Console.WriteLine("Testing NeuralNetwork ReInvented");
            Console.WriteLine("-- From https://www.codeproject.com/Articles/1220276/ReInventing-Neural-Networks");

            NeuralNetwork.NeuralXOR.Main(args);
            Console.ReadLine();
        }
    }
}
