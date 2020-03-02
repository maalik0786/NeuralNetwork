namespace NeuralNetwork
{
	public class MnistData
	{
		public MnistData(int label, float[] image)
		{
			this.label = label;
			this.image = image;
		}

		public readonly float[] image;
		public readonly int label;
	}
}