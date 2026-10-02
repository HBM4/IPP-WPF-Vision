using System.Drawing;

namespace Vision.Core
{
    public class BmpFileReader
    {
        public ColorImage ReadColorBmp(string path)
        {
            // BMP 파일 불러오기
            using Bitmap bitmap = new Bitmap(path);
            int width = bitmap.Width;
            int height = bitmap.Height;

            // 픽셀을 한 칸씩 읽어 R, G, B 채널별 배열에 나눠 담음
            // 32비트 이미지라도 R, G, B 세 값만 쓰고 투명도는 버림
            int pixelCount = width * height;
            byte[] redPixels = new byte[pixelCount];
            byte[] greenPixels = new byte[pixelCount];
            byte[] bluePixels = new byte[pixelCount];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    int index = y * width + x;

                    redPixels[index] = pixel.R;
                    greenPixels[index] = pixel.G;
                    bluePixels[index] = pixel.B;
                }
            }

            return new ColorImage(width, height, redPixels, greenPixels, bluePixels);
        }
    }
}
