using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	public class ImageLoader
	{
		private static IEnumerable<Image> LoadAll(string character) =>
			Directory.GetFiles(Path.Combine(TrainingDataPath, character), "*.png").
				Select(Image.FromFile).ToList();

		private const string TrainingDataPath = @"C:\Code\AiInvestResearch\MnistCustomNeuralNetwork\Optimized16x16";

		private static List<MnistData> FlattenImages(IEnumerable<string> characters)
		{
			var list = new List<MnistData>();
			foreach (string label in characters)
			foreach (var image in LoadAll(label))
				list.Add(new MnistData(int.Parse(label), ConvertImageToInputs(image)));
			return list;
		}

		private static unsafe float[] ConvertImageToInputs(Image image)
		{
			var width = image.Width;
			var height = image.Height;
			var result = new float[width * height];
			var bitmapData = ((Bitmap)image).LockBits(new Rectangle(0, 0, width, height),
				ImageLockMode.ReadWrite, image.PixelFormat);
			int bytesPerPixel = Image.GetPixelFormatSize(image.PixelFormat) / 8;
			var firstPixel = (byte*)bitmapData.Scan0;
			for (int y = 0; y < height; y++)
			{
				var currentLine = firstPixel + (y * bitmapData.Stride);
				for (int x = 0; x < width; x++)
					result[x + y * width] = currentLine[x * bytesPerPixel] / 255.0f;
			}
			((Bitmap)image).UnlockBits(bitmapData);
			return result;
		}

		public Dataset TrainTestSplit(IEnumerable<string> characters)
		{
			var data = FlattenImages(characters);
			List<MnistData> trainInputs = new List<MnistData>();
			List<MnistData> testInputs = new List<MnistData>();
			for (int num = 0; num < data.Count; num++)
				if (num < data.Count * 0.9f)
					trainInputs.Add(new MnistData(data[num].label, data[num].image));
				else
					testInputs.Add(new MnistData(data[num].label, data[num].image));
			return new Dataset(trainInputs, testInputs);
		}
	}
}
