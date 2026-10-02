using System;

namespace Vision.Core
{
    public class Roi
    {
        public int X { get; } // ROI 왼쪽 위 모서리의 가로(X) 좌표
        public int Y { get; } // ROI 왼쪽 위 모서리의 세로(Y) 좌표
        public int Width { get; } // ROI 영역의 폭
        public int Height { get; }  // ROI 영역의 높이
        public bool IsEmpty { get { return Width <= 0 || Height <= 0; } } // 폭 또는 높이가 0 이하인지 확인하여 빈 영역인지 검사
        public static Roi Empty = new Roi(0, 0, 0, 0); // "선택된 영역 없음" 상태를 나타내는 기본 빈 상자

        public Roi(int x, int y, int width, int height) { X = x; Y = y; Width = width; Height = height; } // 생성자
        public static Roi FromPoints(int x1, int y1, int x2, int y2) { return new Roi(Math.Min(x1, x2), Math.Min(y1, y2), Math.Abs(x2 - x1), Math.Abs(y2 - y1)); } // 두 점을 모서리로 하는 ROI를 만듦
    }
}
