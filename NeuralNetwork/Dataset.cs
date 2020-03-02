using System.Collections.Generic;

namespace NeuralNetwork {
	public class Dataset
	{
		public Dataset(List<MnistData> trainInputs, List<MnistData> testInputs)
		{
			train = trainInputs;
			test = testInputs;
		}
		public List<MnistData> train { get; private set; }
		public List<MnistData> test { get; private set; }
	}
}