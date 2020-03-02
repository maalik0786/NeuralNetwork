using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Tests.Datasets
{
    public class HandwrittenDigits
    {
        public void Run()
        {
            Console.WriteLine("Creating neural network...");
						var network = new Model(784, 100, 10, 0.3f);
						var normalizedDataset = File.
							ReadAllLines(
								@"C:\Code\AiInvestResearch\xyzlinearregression\mnist_data\mnist_train.csv").
							Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).
							Select(x =>
								new { Answer = x[0], Inputs = NormalizeInput(x.Skip(1).ToArray())}).
							ToArray();
						Console.WriteLine($"Training network with {normalizedDataset.Length} samples...");
						var s = new Stopwatch();
            s.Start();

            foreach (var input in normalizedDataset.Take(5000))
            {
                var targets = Enumerable.Range(0, 10).Select(x => 0.0f).Select(x => x + 0.01f).ToArray();
                targets[int.Parse(input.Answer)] = 0.99f;
								network.Train(input.Inputs, targets);
            }

            s.Stop();
            Console.WriteLine($"Training complete in {s.ElapsedMilliseconds}ms{Environment.NewLine}");
            Console.WriteLine("Network performance on test data...");
						var queryDataset = File.ReadAllLines(@"C:\Code\AiInvestResearch\xyzlinearregression\mnist_data\mnist_test.csv");
						var queryInputs = queryDataset
                .Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();
						var scoreCard = new List<bool>();

            foreach (var input in queryInputs.Take(100))
            {
                var normalizedInput = NormalizeInput(input.Skip(1).ToArray());
                var correctAnswer = int.Parse(input[0]);
                var response = network.Evaluate(normalizedInput);

                var max = response.Max(x => x);
                var f = response.ToList().IndexOf(max);

                scoreCard.Add(correctAnswer == f);
            }

            Console.WriteLine($"Performed {scoreCard.Count} queries. Correct answers were {scoreCard.Count(x => x)}.");
            Console.WriteLine($"Network has a performance of {scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)}");
        }

				private static float[] NormalizeInput(string[] input) =>
					input.Select(float.Parse).Select(y => (y / 255) * 0.99f + 0.01f).ToArray();
		}
}
