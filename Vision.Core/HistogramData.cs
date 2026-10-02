using System.Linq;

namespace Vision.Core
{
    public class HistogramData
    {
        public int[] Red { get; }
        public int[] Green { get; }
        public int[] Blue { get; }
        public double[] NormalizedRed { get { return Normalize(Red); } } // 최대 빈도를 1로 맞춘 0~1 값
        public double[] NormalizedGreen { get { return Normalize(Green); } }
        public double[] NormalizedBlue { get { return Normalize(Blue); } }

        public HistogramData(int[] red, int[] green, int[] blue) { Red = red; Green = green; Blue = blue; } // 생성자
        private double[] Normalize(int[] counts) // 히스토그램 빈도수 배열을 0~1 사이의 실수 배열로 정규화
        {
            int max = counts.Max();
            double[] values = new double[counts.Length];

            for (int i = 0; i < counts.Length; i++) { values[i] = (double)counts[i] / max; }

            return values;
        }
    }
}
