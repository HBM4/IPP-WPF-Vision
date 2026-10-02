#include "SpatialFilter.h"
#include <immintrin.h>
#include <cfloat>
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

    // byte → float 변환 (AVX2), count는 8의 배수
    void ConvertToFloats(unsigned char* bytes, int count, float* values)
    {
        for (int i = 0; i < count; i += 8)
        {
            __m256i integers = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(bytes + i))); // byte 8개 → int 8개
            _mm256_storeu_ps(values + i, _mm256_cvtepi32_ps(integers)); // int 8개 → float 8개
        }
    }

    // 1차원 컨볼루션 (AVX2): 행마다 출력 8개씩, 4개 누산기에 FMA로 곱더하기
    void ComputeConvolutionAvx2(float* input, int inputStride, int neighborStep, float* kernel, int kernelSize, int columnCount, int rowCount, float* output)
    {
        for (int y = 0; y < rowCount; ++y)
        {
            float* inputRow = input + y * inputStride;
            float* outputRow = output + y * columnCount; // 출력 버퍼의 행 길이 = columnCount

            for (int x = 0; x < columnCount; x += 8) // columnCount가 8의 배수라 나머지 열이 없음
            {
                __m256 sum0 = _mm256_setzero_ps(); // 4개 독립 누산기로 분산
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                int k = 0;
                for (; k + 4 <= kernelSize; k += 4) // 4탭씩 서로 다른 누산기에 FMA로 곱더하기
                {
                    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(inputRow + x + k * neighborStep), _mm256_set1_ps(kernel[k]), sum0);
                    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(inputRow + x + (k + 1) * neighborStep), _mm256_set1_ps(kernel[k + 1]), sum1);
                    sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(inputRow + x + (k + 2) * neighborStep), _mm256_set1_ps(kernel[k + 2]), sum2);
                    sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(inputRow + x + (k + 3) * neighborStep), _mm256_set1_ps(kernel[k + 3]), sum3);
                }
                for (; k < kernelSize; ++k) // 4의 배수로 안 나뉘는 나머지 탭
                {
                    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(inputRow + x + k * neighborStep), _mm256_set1_ps(kernel[k]), sum0);
                }

                _mm256_storeu_ps(outputRow + x, _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3))); // 누산기 4개의 합을 저장
            }
        }
    }

    // DoG 응답 = (좁은 블러 − 넓은 블러) × scale 을 narrowBlur에 덮어씀 (AVX2), count는 8의 배수
    void ComputeDogResponse(float* narrowBlur, float* wideBlur, int count, float scale)
    {
        __m256 scaleVector = _mm256_set1_ps(scale);

        for (int i = 0; i < count; i += 8)
        {
            __m256 difference = _mm256_sub_ps(_mm256_loadu_ps(narrowBlur + i), _mm256_loadu_ps(wideBlur + i));
            _mm256_storeu_ps(narrowBlur + i, _mm256_mul_ps(difference, scaleVector));
        }
    }

    // 소벨 응답 M = (|Gx| + |Gy|) / 2 를 gradientX에 덮어씀 (AVX2), count는 8의 배수
    void ComputeSobelMagnitude(float* gradientX, float* gradientY, int count)
    {
        __m256 signBit = _mm256_set1_ps(-0.0f); // 부호 비트만 1인 값
        __m256 half = _mm256_set1_ps(0.5f);

        for (int i = 0; i < count; i += 8)
        {
            __m256 absoluteX = _mm256_andnot_ps(signBit, _mm256_loadu_ps(gradientX + i)); // |Gx|
            __m256 absoluteY = _mm256_andnot_ps(signBit, _mm256_loadu_ps(gradientY + i)); // |Gy|
            _mm256_storeu_ps(gradientX + i, _mm256_mul_ps(_mm256_add_ps(absoluteX, absoluteY), half));
        }
    }

    // 절댓값·반올림·255 제한 후 byte로 바꿔 ROI에 기록 (AVX2)
    void WriteToPixels(int width, unsigned char* pixels, int roiX, int roiY, int roiWidth, int roiHeight, std::vector<float>& values)
    {
        int paddedWidth = (roiWidth + 7) & ~7; // 결과 버퍼 행 길이 = ROI 폭을 8의 배수로 올린 값
        __m256 signBit = _mm256_set1_ps(-0.0f); // 부호 비트만 1인 값
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 maximum = _mm256_set1_ps(255.0f);

        for (int y = 0; y < roiHeight; ++y)
        {
            float* valueRow = values.data() + y * paddedWidth;
            unsigned char* outputRow = pixels + (roiY + y) * width + roiX; // ROI 안의 y번째 행

            for (int x = 0; x < roiWidth; x += 8) // 값 버퍼 행 길이가 8의 배수라 행 끝에서도 8개를 읽을 수 있음
            {
                __m256 absolute = _mm256_andnot_ps(signBit, _mm256_loadu_ps(valueRow + x)); // 부호 비트를 지워 절댓값
                __m256 clamped = _mm256_min_ps(_mm256_add_ps(absolute, half), maximum); // 반올림용 0.5를 더하고 255 제한
                __m256i integers = _mm256_cvttps_epi32(clamped); // 소수점 버리기 ((Byte) 변환과 동일)
                __m128i shorts = _mm_packus_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1)); // int 8개 → short 8개 (128비트로 나눠야 순서 유지)
                __m128i bytes = _mm_packus_epi16(shorts, shorts); // short 8개 → byte 8개

                if (x + 8 <= roiWidth) { _mm_storel_epi64(reinterpret_cast<__m128i*>(outputRow + x), bytes); } // 8개 모두 기록
                else // 행 끝 8개 미만은 임시 배열에 받은 뒤 필요한 만큼만 복사 (ROI 오른쪽 픽셀 보존)
                {
                    unsigned char last[16];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(last), bytes);
                    std::memcpy(outputRow + x, last, roiWidth - x);
                }
            }
        }
    }
}

#pragma managed(pop)

namespace
{
    // G(x) = exp(-x² / (2σ²))
    std::vector<float> CreateGaussianKernel(int kernelSize, double sigma)
    {
        std::vector<float> kernel(kernelSize);
        int radius = kernelSize / 2;
        double sum = 0.0;

        for (int dx = -radius; dx <= radius; ++dx)
        {
            double value = Math::Exp(-static_cast<double>(dx * dx) / (2.0 * sigma * sigma));

            kernel[dx + radius] = static_cast<float>(value);
            sum += value;
        }

        for (int i = 0; i < kernelSize; ++i)
        {
            kernel[i] = static_cast<float>(kernel[i] / sum); // 총합 1로 정규화(밝기 보존)
            if (kernel[i] < FLT_MIN) { kernel[i] = 0.0f; } // float 비정규 수(1.2e-38 미만)는 0으로: FMA가 수십 배 느려지는 것을 막음
        }

        return kernel;
    }

    // 소벨 = 스무딩 x 미분의 분리형 컨볼루션 (isDerivative=false면 스무딩 커널, true면 미분 커널)
    std::vector<float> CreateSobelKernel(int kernelSize, bool isDerivative)
    {
        std::vector<float> kernel(kernelSize);
        int order = kernelSize - 1; // 이항 계수의 차수
        double coefficient = 1.0;
        std::vector<double> coefficients(order);

        for (int i = 0; i < order; ++i)
        {
            coefficients[i] = coefficient;
            coefficient = coefficient * (order - 1 - i) / (i + 1);
        }

        double sum = 0.0;
        for (int i = 0; i < kernelSize; ++i)
        {
            double left = (i > 0) ? coefficients[i - 1] : 0.0;
            double right = (i < order) ? coefficients[i] : 0.0;

            double value = isDerivative ? (left - right) : (left + right);
            kernel[i] = static_cast<float>(value);

            sum += isDerivative ? Math::Max(value, 0.0) : value; // 미분 커널은 양수 항의 합으로, 스무딩 커널은 전체 합으로 정규화
        }

        for (int i = 0; i < kernelSize; ++i) { kernel[i] = static_cast<float>(kernel[i] / sum); }

        return kernel;
    }

    // DoG 정규화 배율 = 1 / (2차원 DoG 커널 H의 양수 계수 합), 합이 0이면 0 (분모 0 → 응답 0)
    float ComputeDogScale(std::vector<float>& narrowKernel, std::vector<float>& wideKernel)
    {
        int kernelSize = static_cast<int>(narrowKernel.size());
        double positiveSum = 0.0;

        for (int y = 0; y < kernelSize; ++y)
        {
            for (int x = 0; x < kernelSize; ++x)
            {
                double value = static_cast<double>(narrowKernel[y]) * narrowKernel[x] - static_cast<double>(wideKernel[y]) * wideKernel[x]; // H[x, y]
                if (value > 0) { positiveSum += value; }
            }
        }

        if (positiveSum <= 0) { return 0.0f; } // σ가 아주 작으면 두 커널이 float에서 같아져 양수 항이 없음
        return static_cast<float>(1.0 / positiveSum);
    }

    // ROI를 커널 반경만큼 넓힌 float 확장 이미지
    std::vector<float> CreateExtendedImage(int width, int height, unsigned char* pixels, int roiX, int roiY, int roiWidth, int roiHeight, int kernelSize)
    {
        int radius = kernelSize / 2;
        int paddedWidth = (roiWidth + 7) & ~7; // 결과 폭 = ROI 폭을 8의 배수로 올린 값 (AVX2가 8개씩 처리)
        int extendedStride = (paddedWidth + 2 * radius + 7) & ~7; // 확장 행 길이 = 결과 폭 + 양옆 반경을 8의 배수로 올린 값
        int count = extendedStride * (roiHeight + 2 * radius); // 8의 배수
        std::vector<unsigned char> extendedBytes(count); // 행 끝 여유 칸은 0
        std::vector<float> extended(count);

        FillExtendedImage(pixels, width, height, roiX, roiY, roiWidth, roiHeight, radius, extendedBytes.data(), extendedStride);
        ConvertToFloats(extendedBytes.data(), count, extended.data());

        return extended;
    }

    // 가로 커널 → 세로 커널 순서로 1D 컨볼루션
    std::vector<float> ComputeSeparableConvolution(int roiWidth, int roiHeight, std::vector<float>& extended, std::vector<float>& horizontalKernel, std::vector<float>& verticalKernel)
    {
        int kernelSize = static_cast<int>(horizontalKernel.size());
        int radius = kernelSize / 2;
        int paddedWidth = (roiWidth + 7) & ~7; // 결과 행 길이 = ROI 폭을 8의 배수로 올린 값
        int extendedStride = (paddedWidth + 2 * radius + 7) & ~7; // 확장 행 길이 (CreateExtendedImage와 같은 계산)
        int extendedHeight = roiHeight + 2 * radius;
        std::vector<float> horizontal(paddedWidth * extendedHeight); // 가로 방향 컨볼루션의 중간 결과
        std::vector<float> vertical(paddedWidth * roiHeight); // 세로 방향까지 마친 결과

        ComputeConvolutionAvx2(extended.data(), extendedStride, 1, horizontalKernel.data(), kernelSize, paddedWidth, extendedHeight, horizontal.data()); // 확장된 입력에 가로 커널
        ComputeConvolutionAvx2(horizontal.data(), paddedWidth, paddedWidth, verticalKernel.data(), kernelSize, paddedWidth, roiHeight, vertical.data()); // 가로 결과에 세로 커널

        return vertical;
    }
}

namespace Vision::CppCli
{
    void SpatialFilter::Gaussian(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize, double sigma)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        std::vector<float> kernel = CreateGaussianKernel(kernelSize, sigma); // 가로/세로에 동일하게 적용되는 1차원 가우시안 커널

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열
            pin_ptr<Byte> pixelPointer = &pixels[0]; // 네이티브 함수에 넘기도록 배열 주소를 한 번만 고정

            std::vector<float> extended = CreateExtendedImage(image->Width, image->Height, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, kernelSize);
            std::vector<float> result = ComputeSeparableConvolution(roi->Width, roi->Height, extended, kernel, kernel); // 가로/세로로 블러링

            WriteToPixels(image->Width, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, result);
        }
    }
    void SpatialFilter::Laplacian(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize, double sigma)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        std::vector<float> narrowKernel = CreateGaussianKernel(kernelSize, sigma); // 원래 시그마 가우시안 커널
        std::vector<float> wideKernel = CreateGaussianKernel(kernelSize, sigma * 1.6); // 더 넓은 시그마 가우시안 커널
        float scale = ComputeDogScale(narrowKernel, wideKernel); // 2D DoG 커널의 양수 합으로 정규화하는 배율

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열
            pin_ptr<Byte> pixelPointer = &pixels[0]; // 네이티브 함수에 넘기도록 배열 주소를 한 번만 고정

            std::vector<float> extended = CreateExtendedImage(image->Width, image->Height, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, kernelSize);
            std::vector<float> narrowBlur = ComputeSeparableConvolution(roi->Width, roi->Height, extended, narrowKernel, narrowKernel); // 원래 시그마로 블러링
            std::vector<float> wideBlur = ComputeSeparableConvolution(roi->Width, roi->Height, extended, wideKernel, wideKernel); // 더 넓은 시그마로 블러링
            ComputeDogResponse(narrowBlur.data(), wideBlur.data(), static_cast<int>(narrowBlur.size()), scale); // DoG = (narrowBlur - wideBlur) × 배율

            WriteToPixels(image->Width, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, narrowBlur);
        }
    }
    void SpatialFilter::Sobel(ColorImage^ image, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize)
    {
        if (roi->IsEmpty) { roi = gcnew Roi(0, 0, image->Width, image->Height); } // 유효하지 않은 ROI는 전체 이미지 영역으로 처리

        std::vector<float> smoothKernel = CreateSobelKernel(kernelSize, false); // 스무딩 커널
        std::vector<float> derivativeKernel = CreateSobelKernel(kernelSize, true); // 미분 커널

        array<array<Byte>^>^ channelPixels = gcnew array<array<Byte>^> { image->RedPixels, image->GreenPixels, image->BluePixels }; // R, G, B 배열
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        #pragma omp parallel for num_threads(3) // R, G, B를 스레드 3개가 하나씩 계산 (남는 스레드가 CPU를 잡아 느려지지 않도록 3개만)
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }
            array<Byte>^ pixels = channelPixels[channel]; // 이 채널의 픽셀 배열
            pin_ptr<Byte> pixelPointer = &pixels[0]; // 네이티브 함수에 넘기도록 배열 주소를 한 번만 고정

            std::vector<float> extended = CreateExtendedImage(image->Width, image->Height, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, kernelSize);
            std::vector<float> gradientX = ComputeSeparableConvolution(roi->Width, roi->Height, extended, derivativeKernel, smoothKernel); // 좌우 밝기 변화 = 세로 엣지 검출
            std::vector<float> gradientY = ComputeSeparableConvolution(roi->Width, roi->Height, extended, smoothKernel, derivativeKernel); // 상하 밝기 변화 = 가로 엣지 검출
            ComputeSobelMagnitude(gradientX.data(), gradientY.data(), static_cast<int>(gradientX.size())); // M = (|Gx| + |Gy|) / 2

            WriteToPixels(image->Width, pixelPointer, roi->X, roi->Y, roi->Width, roi->Height, gradientX);
        }
    }
}
