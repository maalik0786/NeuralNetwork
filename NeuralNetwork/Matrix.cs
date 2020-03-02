using System;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class Matrix
	{
		private Matrix(int rows, int cols)
		{
			value = new float[rows][];
			for (var i = 0; i < rows; i++)
				value[i] = new float[cols];
		}

		private Matrix(float[][] array) => value = array;
		public float[][] value { get; }


		private static float[][] CreateJagged(int rows, int cols)
		{
			var jagged = new float[rows][];
			for (var i = 0; i < rows; i++)
				jagged[i] = new float[cols];
			return jagged;
		}

		public static Matrix Create(int rows, int cols) => new Matrix(rows, cols);
		public static Matrix Create(float[][] array) => new Matrix(array);

		public void Initialize(Func<float> elementInitializer)
		{
			for (var x = 0; x < value.Length; x++)
			for (var y = 0; y < value[x].Length; y++)
				value[x][y] = elementInitializer();
		}

		public static Matrix operator -(Matrix a, Matrix b)
		{
			var newMatrix = CreateJagged(a.value.Length, b.value[0].Length);
			for (var x = 0; x < a.value.Length; x++)
			for (var y = 0; y < a.value[x].Length; y++)
				newMatrix[x][y] = a.value[x][y] - b.value[x][y];
			return Create(newMatrix);
		}

		public static Matrix operator +(Matrix a, Matrix b)
		{
			var newMatrix = CreateJagged(a.value.Length, b.value[0].Length);
			for (var x = 0; x < a.value.Length; x++)
			for (var y = 0; y < a.value[x].Length; y++)
				newMatrix[x][y] = a.value[x][y] + b.value[x][y];
			return Create(newMatrix);
		}

		public static Matrix operator +(Matrix a, float b)
		{
			for (var x = 0; x < a.value.Length; x++)
			for (var y = 0; y < a.value[x].Length; y++)
				a.value[x][y] = a.value[x][y] + b;
			return a;
		}

		public static Matrix operator -(float a, Matrix m)
		{
			for (var x = 0; x < m.value.Length; x++)
			for (var y = 0; y < m.value[x].Length; y++)
				m.value[x][y] = a - m.value[x][y];
			return m;
		}

		public static Matrix operator *(Matrix a, Matrix b)
		{
			if (a.value.Length == b.value.Length && a.value[0].Length == b.value[0].Length)
			{
				var m = CreateJagged(a.value.Length, a.value[0].Length);
				Parallel.For(0, a.value.Length, i =>
				{
					for (var j = 0; j < a.value[i].Length; j++)
						m[i][j] = a.value[i][j] * b.value[i][j];
				});
				return Create(m);
			}

			var newMatrix = CreateJagged(a.value.Length, b.value[0].Length);
			if (a.value[0].Length == b.value.Length)
			{
				Parallel.For(0, a.value.Length, i =>
				{
					for (var j = 0; j < b.value[0].Length; j++)
					{
						var temp = 0.0f;
						for (var k = 0; k < a.value[0].Length; k++)
							temp += a.value[i][k] * b.value[k][j];
						newMatrix[i][j] = temp;
					}
				});
			}

			return Create(newMatrix);
		}

		public static Matrix operator *(float scalar, Matrix b)
		{
			var newMatrix = CreateJagged(b.value.Length, b.value[0].Length);
			for (var x = 0; x < b.value.Length; x++)
			for (var y = 0; y < b.value[x].Length; y++)
				newMatrix[x][y] = b.value[x][y] * scalar;
			return Create(newMatrix);
		}

		public Matrix Transpose()
		{
			var rows = value.Length;
			var newMatrix = CreateJagged(value[0].Length, rows);
			for (var row = 0; row < rows; row++)
			for (var col = 0; col < value[row].Length; col++)
				newMatrix[col][row] = value[row][col];
			return Create(newMatrix);
		}
	}
}