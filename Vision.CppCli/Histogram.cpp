#include "Histogram.h"

using namespace System;
using namespace Vision::Core;

namespace
{
    // 채널 ROI 영역에 대한 히스토그램 데이터를 계산함
    array<int>^ CalculateChannelHistogram(int width, array<Byte>^ pixels, int roiX, int roiY, int roiWidth, int roiHeight)
    {
        array<int>^ histogram = gcnew array<int>(256);

        for (int y = roiY; y < roiY + roiHeight; ++y)
        {
            for (int x = roiX; x < roiX + roiWidth; ++x)
            {
                histogram[pixels[y * width + x]] += 1;
            }
        }

        return histogram;
    }

    // 히스토그램 분포에서 그룹 간 분산이 최대가 되는 Otsu 전역 임계값을 탐색
    int CalculateThreshold(array<int>^ histogram, long long pixelCount)
    {
        double intensityTotal = 0;

        for (int value = 0; value < 256; ++value)
        {
            intensityTotal += static_cast<double>(value) * histogram[value]; // S = Σ(i × h(i)); double로 오버플로우 방지
        }

        int bestThreshold = 0;
        double bestVariance = -1.0;
        long long darkPixelCount = 0;
        double darkIntensitySum = 0;

        for (int t = 0; t < 255; ++t)
        {
            darkPixelCount += histogram[t]; // n_d = Σh(i), i ≤ t
            darkIntensitySum += static_cast<double>(t) * histogram[t]; // S_d = Σ(i × h(i)), i ≤ t
            long long brightPixelCount = pixelCount - darkPixelCount; // n_b = N - n_d

            if (darkPixelCount == 0 || brightPixelCount == 0) { continue; } // 한쪽 그룹이 비어 있으면 건너뜀

            double brightIntensitySum = intensityTotal - darkIntensitySum; // S_b = S - S_d
            double meanDelta = darkIntensitySum / darkPixelCount - brightIntensitySum / brightPixelCount; // Δμ = μ_d - μ_b
            double betweenVariance = static_cast<double>(darkPixelCount) * brightPixelCount * meanDelta * meanDelta; // σ² = n_d × n_b × (Δμ)²

            // 그룹 간 분산이 가장 큰 임계값을 선택 (동점이면 먼저 찾은 t를 유지)
            if (betweenVariance > bestVariance)
            {
                bestVariance = betweenVariance;
                bestThreshold = t;
            }
        }

        return bestThreshold;
    }

    // 직접 누적분포함수(CDF)로 명암도 변환 LUT를 계산함 (식 3-15: s_k = (L-1) × Σ(j=0..k) p_r(r_j))
    array<Byte>^ CalculateLookupTable(array<int>^ histogram, long long pixelCount)
    {
        array<Byte>^ lookupTable = gcnew array<Byte>(256);
        long long cumulativeCount = 0;

        for (int k = 0; k < 256; ++k)
        {
            cumulativeCount += histogram[k]; // C[k] = Σh[j], j = 0..k

            // s_k = round(255 × C[k] / N); 정수 나눗셈에 N/2를 더해 반올림 처리
            lookupTable[k] = static_cast<Byte>((cumulativeCount * 255 + pixelCount / 2) / pixelCount);
        }

        return lookupTable;
    }

    // 임계값보다 크면 255, 아니면 0을 ROI에 기록
    void ApplyThreshold(int width, array<Byte>^ pixels, int roiX, int roiY, int roiWidth, int roiHeight, int threshold)
    {
        for (int y = roiY; y < roiY + roiHeight; ++y)
        {
            for (int x = roiX; x < roiX + roiWidth; ++x)
            {
                int index = y * width + x;
                pixels[index] = pixels[index] > threshold ? 255 : 0;
            }
        }
    }

    // LUT로 바꾼 명암값을 ROI에 기록
    void ApplyLookupTable(int width, array<Byte>^ pixels, int roiX, int roiY, int roiWidth, int roiHeight, array<Byte>^ lookupTable)
    {
        for (int y = roiY; y < roiY + roiHeight; ++y)
        {
            for (int x = roiX; x < roiX + roiWidth; ++x)
            {
                int index = y * width + x;
                pixels[index] = lookupTable[pixels[index]];
            }
        }
    }
}

namespace Vision::CppCli
{
    HistogramData^ Histogram::Calculate(ColorImage^ image, Roi^ roi)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        return gcnew HistogramData(
            CalculateChannelHistogram(image->Width, image->RedPixels, roi->X, roi->Y, roi->Width, roi->Height),
            CalculateChannelHistogram(image->Width, image->GreenPixels, roi->X, roi->Y, roi->Width, roi->Height),
            CalculateChannelHistogram(image->Width, image->BluePixels, roi->X, roi->Y, roi->Width, roi->Height));
    }
    void Histogram::Binarize(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열

            array<int>^ histogram = CalculateChannelHistogram(image->Width, pixels, roi->X, roi->Y, roi->Width, roi->Height);
            int threshold = CalculateThreshold(histogram, static_cast<long long>(roi->Width) * roi->Height); // ROI 넓이 계산도 int 곱셈으로 먼저 넘치지 않도록 64비트로 계산

            ApplyThreshold(image->Width, pixels, roi->X, roi->Y, roi->Width, roi->Height, threshold);
        }
    }
    void Histogram::Equalize(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열

            array<int>^ histogram = CalculateChannelHistogram(image->Width, pixels, roi->X, roi->Y, roi->Width, roi->Height);
            array<Byte>^ lookupTable = CalculateLookupTable(histogram, static_cast<long long>(roi->Width) * roi->Height); // ROI 넓이 계산도 int 곱셈으로 먼저 넘치지 않도록 64비트로 계산

            ApplyLookupTable(image->Width, pixels, roi->X, roi->Y, roi->Width, roi->Height, lookupTable);
        }
    }
}
