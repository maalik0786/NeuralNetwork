using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetwork.Tests.Datasets
{
	public class MnistDigits
	{
		public void Run()
		{
			Console.WriteLine("Creating neural network...");
			var network = new Model(256, 7, 4, 0.3f);
			var dataset =
				new ImageLoader().TrainTestSplit(new[] { "0", "1", "2", "3" });
			var s = new Stopwatch();
			s.Start();
			foreach (var input in dataset.train)
			{
				var targets = Enumerable.Range(0, 4).Select(x => 0.0f).Select(x => x + 0.01f).
					ToArray();
				targets[input.label] = 0.99f;
				network.Train(input.image, targets);
			}

			s.Stop();
			Console.WriteLine(
				$"Training complete in {s.ElapsedMilliseconds}ms{Environment.NewLine}");

			var scoreCard = new List<bool>();
			foreach (var input in dataset.test)
			{
				var response = network.Evaluate(input.image);
				int prediction = response.ToList().IndexOf(response.Max(x => x));
				scoreCard.Add(input.label == prediction);
			}

			Console.WriteLine(
				$"Performed {scoreCard.Count} queries. Correct answers were {scoreCard.Count(x => x)}.");
			Console.WriteLine(
				$"Network has a performance of {scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)}");
		}
	}
}
