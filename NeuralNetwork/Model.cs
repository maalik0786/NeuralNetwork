using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
	public class Model
	{
		private readonly float learningRate;
		private Matrix weightHiddenOutput;
		private Matrix weightInputHidden;

		public Model(int numberOfInputNodes, int numberOfHiddenNodes,
			int numberOfOutputNodes, float learningRate)
		{
			this.learningRate = learningRate;
			weightInputHidden = Matrix.Create(numberOfHiddenNodes, numberOfInputNodes);
			weightHiddenOutput = Matrix.Create(numberOfOutputNodes, numberOfHiddenNodes);
			RandomizeWeights();
		}

		/// <summary>
		/// Random weights are distributed -0.5 to 0.5
		/// </summary>
		private void RandomizeWeights()
		{
			var rnd = new Random();
			weightHiddenOutput.Initialize(() => (float) rnd.NextDouble() - 0.5f);
			weightInputHidden.Initialize(() => (float) rnd.NextDouble() - 0.5f);
		}

		public void Train(float[] inputs, float[] targets)
		{
			var inputSignals = ConvertToMatrix(inputs);
			var targetSignals = ConvertToMatrix(targets);
			var hiddenOutputs = Sigmoid(weightInputHidden * inputSignals);
			var finalOutputs = Sigmoid(weightHiddenOutput * hiddenOutputs);
			var outputErrors = targetSignals - finalOutputs;
			var hiddenErrors = weightHiddenOutput.Transpose() * outputErrors;
			weightHiddenOutput += learningRate * outputErrors * finalOutputs *
				(1.0f - finalOutputs) * hiddenOutputs.Transpose();
			weightInputHidden += learningRate * hiddenErrors * hiddenOutputs *
				(1.0f - hiddenOutputs) * inputSignals.Transpose();
		}

		public float[] Evaluate(float[] inputs)
		{
			var inputSignals = ConvertToMatrix(inputs);
			var hiddenOutputs = Sigmoid(weightInputHidden * inputSignals);
			var finalOutputs = Sigmoid(weightHiddenOutput * hiddenOutputs);
			return finalOutputs.value.SelectMany(x => x.Select(y => y)).ToArray();
		}

		private static Matrix ConvertToMatrix(IReadOnlyList<float> inputList)
		{
			var input = new float[inputList.Count][];
			for (var x = 0; x < input.Length; x++)
				input[x] = new[] { inputList[x] };
			return Matrix.Create(input);
		}

		private static Matrix Sigmoid(Matrix matrix)
		{
			var newMatrix = Matrix.Create(matrix.value.Length, matrix.value[0].Length);
			for (var x = 0; x < matrix.value.Length; x++)
			for (var y = 0; y < matrix.value[x].Length; y++)
				newMatrix.value[x][y] = (float)(1 / (1 + Math.Pow(Math.E, -matrix.value[x][y])));
			return newMatrix;
		}
	}
}