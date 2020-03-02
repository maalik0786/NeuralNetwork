using NeuralNetwork.Tests.Datasets;
using NUnit.Framework;

namespace NeuralNetwork.Tests
{
	internal class NeuralNetworkTests
	{
		[Test, Category("Slow")]
		public void MnistDigits() => new MnistDigits().Run();

		[Test, Category("Slow")]
		public void Xor() => new Xor().Run();

		[Test, Category("Slow")]
		public void Sine() => new Sine().Run();

		[Test]
		public void Iris() => new Iris().Run();

		[Test, Category("Slow")]
		public void HandWrittenDigits() => new HandwrittenDigits().Run();
	}
}
