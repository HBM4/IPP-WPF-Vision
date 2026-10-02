namespace Vision.Core
{
    public class ColorImage
    {
        public int Width { get; }
        public int Height { get; }
        public byte[] RedPixels { get; }
        public byte[] GreenPixels { get; }
        public byte[] BluePixels { get; }

        public ColorImage(int width, int height, byte[] redPixels, byte[] greenPixels, byte[] bluePixels) { Width = width; Height = height; RedPixels = redPixels; GreenPixels = greenPixels; BluePixels = bluePixels; } // 생성자
        public ColorImage Clone() { return new ColorImage(Width, Height, (byte[])RedPixels.Clone(), (byte[])GreenPixels.Clone(), (byte[])BluePixels.Clone()); } // 세 채널 배열을 모두 복제하는 새 이미지
        public byte[] ToBgrBytes() // R, G, B 배열을 픽셀당 B, G, R 세 바이트로 이어 붙임 (Bgr24 비트맵용)
        {
            int pixelCount = Width * Height;
            byte[] bgrPixels = new byte[pixelCount * 3];

            for (int i = 0; i < pixelCount; i++)
            {
                bgrPixels[i * 3] = BluePixels[i];
                bgrPixels[i * 3 + 1] = GreenPixels[i];
                bgrPixels[i * 3 + 2] = RedPixels[i];
            }

            return bgrPixels;
        }
    }
}
