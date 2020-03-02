using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Tests.Datasets
{
	public class Iris
	{
		private static readonly List<string> PossibleResults =
			new List<string> { "setosa", "versicolor", "virginica" };

		public void Run()
		{
			var shuffledInputs = GetInputs();
			var network = new Model(4, 5, 3, 0.2f);
			var trainDataSet = shuffledInputs.Take(100).ToArray();
			const int Epochs = 500;
			Console.WriteLine(
				$"Training network with {trainDataSet.Length} samples using {Epochs} epochs...");
			for (var epoch = 0; epoch < Epochs; epoch++)
				foreach (var input in trainDataSet)
				{
					var targets = new[] { 0.01f, 0.01f, 0.01f };
					targets[PossibleResults.IndexOf(input.Last())] = 0.99f;
					var inputList = input.Take(4).Select(float.Parse).ToArray();
					network.Train(NormalizeIrisData(inputList), targets);
				}

			var scoreCard = new List<bool>();
			var testDataset = shuffledInputs.Skip(100).ToArray();
			foreach (var input in testDataset)
			{
				var result = network.
					Evaluate(NormalizeIrisData(input.Take(4).Select(float.Parse).ToArray())).ToList();
				var answer = PossibleResults[PossibleResults.IndexOf(input.Last())];
				var predictedIris = PossibleResults[result.IndexOf(result.Max())];
				scoreCard.Add(answer == predictedIris);
			}

			Console.WriteLine(
				$"Performance is {scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count) * 100} percent.");
		}

		private static string[][] GetInputs()
		{
			var dataset = File.ReadAllLines(@"C:\Code\AiInvestResearch\xyzlinearregression\Iris_data\iris.csv");
			var allInputs = dataset.
				Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();
			var shuffledInputs = Shuffle(allInputs);
			return shuffledInputs;
		}

		private static string[][] Shuffle(string[][] allInputs) =>
			allInputs.OrderBy(x => new Random().NextDouble()).ToArray();

		private static float[] NormalizeIrisData(float[] input)
		{
			var maxSepalLength = 7.9f;
			var maxSepalWidth = 4.4f;
			var maxPetalLength = 6.9f;
			var maxPetalWidth = 2.5f;
			var normalized = new[]
			{
				input[0] / maxSepalLength + 0.01f, input[1] / maxSepalWidth + 0.01f,
				input[2] / maxPetalLength + 0.01f, input[3] / maxPetalWidth + 0.01f
			};
			return normalized;
		}
	}
}