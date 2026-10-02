#include "Morphology.h"
#include <immintrin.h>
#include <cstring>
#include <vector>

using namespace System;
using namespace Vision::Core;

#pragma managed(push, off)

namespace
{
    // ROI를 커널 반경만큼 넓혀 복사하고, 이미지 밖은 가장 가까운 끝 픽셀로 채움 (행마다 memset·memcpy)
    void FillExtendedImage(unsigned char* pixels, int width, int height, int roiX, int roiY, int roiWidth, int roiHeight, int radius, unsigned char* extended, int extendedStride)
    {
        int extendedWidth = roiWidth + 2 * radius; // 값을 채울 폭 (그 뒤 여유 칸은 0으로 둠)
        int startX = roiX - radius; // 확장 이미지 첫 열에 해당하는 원본 열
        int leftCount = startX < 0 ? -startX : 0; // 왼쪽 이미지 밖 열 수
        int insideEnd = width - startX < extendedWidth ? width - startX : extendedWidth; // 오른쪽 이미지 밖이 시작되는 확장 열

        for (int ey = 0; ey < roiHeight + 2 * radius; ++ey)
        {
            int sourceY = roiY + ey - radius;
            if (sourceY < 0) { sourceY = 0; } // 위쪽 밖은 첫 행으로 클램프
            if (sourceY > height - 1) { sourceY = height - 1; } // 아래쪽 밖은 마지막 행으로 클램프

            unsigned char* sourceRow = pixels + sourceY * width; // 복사할 원본 행
            unsigned char* extendedRow = extended + ey * extendedStride; // 채울 확장 행
            std::memset(extendedRow, sourceRow[0], leftCount); // 왼쪽 밖은 첫 열 값
            std::memcpy(extendedRow + leftCount, sourceRow + startX + leftCount, insideEnd - leftCount); // 이미지 안은 그대로 복사
            std::memset(extendedRow + insideEnd, sourceRow[width - 1], extendedWidth - insideEnd); // 오른쪽 밖은 마지막 열 값
        }
    }

    // 행마다 32개씩 kernelSize개 이웃의 최댓값(팽창) 또는 최솟값(수축)을 구해 columnCount개를 기록 (AVX2)
    void ComputeMinMaxAvx2(unsigned char* input, int inputStride, int neighborStep, int kernelSize, bool useMaximum, int columnCount, int rowCount, unsigned char* output, int outputStride)
    {
        for (int y = 0; y < rowCount; ++y)
        {
            unsigned char* inputRow = input + y * inputStride;
            unsigned char* outputRow = output + y * outputStride;

            for (int x = 0; x < columnCount; x += 32) // 입력 행 길이를 32의 배수 이상으로 잡아 두어 행 끝에서도 32개를 읽을 수 있음
            {
                __m256i selected = _mm256_loadu_si256(reinterpret_cast<__m256i*>(inputRow + x)); // 첫 이웃에서 시작

                for (int k = 1; k < kernelSize; ++k) // 나머지 이웃을 하나씩 병합
                {
                    __m256i neighbor = _mm256_loadu_si256(reinterpret_cast<__m256i*>(inputRow + x + k * neighborStep));
                    selected = useMaximum ? _mm256_max_epu8(selected, neighbor) : _mm256_min_epu8(selected, neighbor); // 팽창은 최댓값, 수축은 최솟값
                }

                if (x + 32 <= columnCount) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputRow + x), selected); } // 32개 모두 기록
                else // 행 끝 32개 미만은 임시 배열에 받은 뒤 필요한 만큼만 복사 (ROI 오른쪽 픽셀 보존)
                {
                    unsigned char last[32];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(last), selected);
                    std::memcpy(outputRow + x, last, columnCount - x);
                }
            }
        }
    }
}

#pragma managed(pop)

namespace
{
    // 채널 하나를 가로 min/max → 세로 min/max 순서로 계산해 ROI에 기록 (중간 버퍼는 네이티브, 배열은 한 번만 고정)
    void ComputeSeparableMinMax(int width, int height, array<Byte>^ pixels, int roiX, int roiY, int roiWidth, int roiHeight, int kernelSize, bool useMaximum)
    {
        int radius = kernelSize / 2;
        int extendedHeight = roiHeight + 2 * radius; // 위아래로 반경만큼 확장한 행 수
        int paddedWidth = (roiWidth + 31) & ~31; // 가로 결과 행 길이: ROI 폭을 32의 배수로 올림 (세로 계산이 행 끝에서도 32개를 읽도록)
        int extendedStride = paddedWidth + 2 * radius; // 확장 이미지 행 길이: 가로 계산이 행 끝에서 32개를 읽어도 행 안에 머묾

        std::vector<unsigned char> extended(extendedStride * extendedHeight); // 경계를 확장한 입력 (여유 칸은 0)
        std::vector<unsigned char> horizontal(paddedWidth * extendedHeight); // 가로 min/max 결과 (여유 칸은 0)

        pin_ptr<Byte> pixelPointer = &pixels[0]; // 네이티브 함수에 넘기도록 배열 주소를 한 번만 고정
        unsigned char* pixelStart = pixelPointer; // 고정한 배열의 시작 주소

        FillExtendedImage(pixelStart, width, height, roiX, roiY, roiWidth, roiHeight, radius, extended.data(), extendedStride);
        ComputeMinMaxAvx2(extended.data(), extendedStride, 1, kernelSize, useMaximum, roiWidth, extendedHeight, horizontal.data(), paddedWidth); // 가로 방향
        ComputeMinMaxAvx2(horizontal.data(), paddedWidth, paddedWidth, kernelSize, useMaximum, roiWidth, roiHeight, pixelStart + roiY * width + roiX, width); // 세로 방향 → ROI에 바로 기록
    }
}

namespace Vision::CppCli
{
    void Morphology::Dilate(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열

            ComputeSeparableMinMax(image->Width, image->Height, pixels, roi->X, roi->Y, roi->Width, roi->Height, kernelSize, true); // 가로/세로 Max = 팽창
        }
    }
    void Morphology::Erode(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열

            ComputeSeparableMinMax(image->Width, image->Height, pixels, roi->X, roi->Y, roi->Width, roi->Height, kernelSize, false); // 가로/세로 Min = 수축
        }
    }
}
