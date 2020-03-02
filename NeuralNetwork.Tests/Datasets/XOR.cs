using System;

namespace NeuralNetwork.Tests.Datasets
{
	public class Xor
	{
		public void Run()
		{
			Console.WriteLine("Creating neural network...");
			var network = new Model(2, 5, 1, 1.8f);
			for (var epoch = 0; epoch < 2000; epoch++)
			{
				network.Train(new[] { 0.01f, 0.01f }, new[] { 0.01f });
				network.Train(new[] { 0.01f, 0.99f }, new[] { 0.99f });
				network.Train(new[] { 0.99f, 0.01f }, new[] { 0.99f });
				network.Train(new[] { 0.99f, 0.99f }, new[] { 0.01f });
			}

			var true1 = network.Evaluate(new[] { 0.99f, 0.01f });
			var true2 = network.Evaluate(new[] { 0.01f, 0.99f });
			var false1 = network.Evaluate(new[] { 0.01f, 0.01f });
			var false2 = network.Evaluate(new[] { 0.99f, 0.99f });
			Console.WriteLine($"Networks answer for true_1: {true1[0]} which is {true1[0] > 0.5}");
			Console.WriteLine($"Networks answer for true_2: {true2[0]} which is {true2[0] > 0.5}");
			Console.WriteLine(
				$"Networks answer for false_1: {false1[0]} which is {false1[0] > 0.5}");
			Console.WriteLine(
				$"Networks answer for false_2: {false2[0]} which is {false2[0] > 0.5}");
		}
	}
}