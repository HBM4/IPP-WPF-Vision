#include "TemplateMatching.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace System;
using namespace System::Collections::Generic;
using namespace Vision::Core;

#pragma managed(push, off)

namespace
{
    // 벡터의 32비트 칸 8개를 모두 더함
    long long SumVectorValues(__m256i values)
    {
        int parts[8];

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(parts), values); // 벡터를 일반 배열로 꺼냄

        long long sum = 0;
        for (int i = 0; i < 8; ++i) { sum += parts[i]; } // 8칸을 하나씩 더함

        return sum;
    }

    // 후보 영역의 세 합을 AVX2로 계산해 candidateSum(ΣC), candidateSquaredSum(ΣC²), productSum(ΣT×C)에 채움
    void ComputeSumsAvx2(unsigned char* templateStart, unsigned char* candidateStart, int resultWidth, int roiWidth, int roiHeight, long long& candidateSum, long long& candidateSquaredSum, long long& productSum)
    {
        __m256i zero = _mm256_setzero_si256(); // 모든 칸이 0인 벡터
        __m256i candidateSumVector = zero; // ΣC 누산기
        __m256i candidateSquaredSumVector = zero; // ΣC² 누산기
        __m256i productSumVector = zero; // ΣT×C 누산기

        int remainder = roiWidth % 16; // 16개씩 나누고 남는 픽셀 수
        __m128i byteIndex = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); // 칸 번호 0~15
        __m128i lastBlockMask = _mm_cmpgt_epi8(byteIndex, _mm_set1_epi8(static_cast<char>(15 - remainder))); // 뒤쪽 remainder칸만 남기는 마스크

        int blockCountPerRow = (roiWidth + 15) / 16; // 한 행의 16픽셀 블록 수 (올림)
        int addedBlockCount = 0; // 32비트 누산기에 더한 블록 수

        candidateSquaredSum = 0; // 64비트 합은 0에서 시작
        productSum = 0;
        for (int y = 0; y < roiHeight; ++y) // ROI의 각 행
        {
            unsigned char* templateRow = templateStart + y * roiWidth; // 템플릿 현재 행의 시작 주소 (템플릿은 ROI만 담으므로 행 길이 = ROI 폭)
            unsigned char* candidateRow = candidateStart + y * resultWidth; // 후보 현재 행의 시작 주소
            int x = 0; // 현재 행에서 처리한 픽셀 수

            // ① 32개씩 처리 ───────────────────────────────────────────────────────
            while (x + 32 <= roiWidth)
            {
                __m256i templateBytes = _mm256_loadu_si256(reinterpret_cast<__m256i*>(templateRow + x)); // 템플릿 픽셀 32개 읽기
                __m256i candidateBytes = _mm256_loadu_si256(reinterpret_cast<__m256i*>(candidateRow + x)); // 후보 픽셀 32개 읽기

                // 곱하기 전에 8비트를 16비트로 늘림 (순서가 섞여도 합은 같음)
                __m256i templateLow = _mm256_unpacklo_epi8(templateBytes, zero);
                __m256i templateHigh = _mm256_unpackhi_epi8(templateBytes, zero);
                __m256i candidateLow = _mm256_unpacklo_epi8(candidateBytes, zero);
                __m256i candidateHigh = _mm256_unpackhi_epi8(candidateBytes, zero);

                __m256i candidateSumPart = _mm256_sad_epu8(candidateBytes, zero); // C를 8개씩 더함
                __m256i squaresLow = _mm256_madd_epi16(candidateLow, candidateLow); // C×C를 2개씩 더함
                __m256i squaresHigh = _mm256_madd_epi16(candidateHigh, candidateHigh);
                __m256i productsLow = _mm256_madd_epi16(templateLow, candidateLow); // T×C를 2개씩 더함
                __m256i productsHigh = _mm256_madd_epi16(templateHigh, candidateHigh);

                candidateSumVector = _mm256_add_epi64(candidateSumVector, candidateSumPart); // ΣC에 더함
                candidateSquaredSumVector = _mm256_add_epi32(candidateSquaredSumVector, _mm256_add_epi32(squaresLow, squaresHigh)); // ΣC²에 더함
                productSumVector = _mm256_add_epi32(productSumVector, _mm256_add_epi32(productsLow, productsHigh)); // ΣT×C에 더함
                x += 32;
            }

            // ② 남은 16개 처리 ──────────────────────────────────────────────────────
            if (x + 16 <= roiWidth)
            {
                __m128i templateBytes = _mm_loadu_si128(reinterpret_cast<__m128i*>(templateRow + x)); // 템플릿 픽셀 16개 읽기
                __m128i candidateBytes = _mm_loadu_si128(reinterpret_cast<__m128i*>(candidateRow + x)); // 후보 픽셀 16개 읽기
                __m256i templateValues = _mm256_cvtepu8_epi16(templateBytes); // 8비트 → 16비트
                __m256i candidateValues = _mm256_cvtepu8_epi16(candidateBytes);

                __m256i candidateSumPart = _mm256_sad_epu8(_mm256_zextsi128_si256(candidateBytes), zero); // C를 8개씩 더함
                candidateSumVector = _mm256_add_epi64(candidateSumVector, candidateSumPart); // ΣC에 더함
                candidateSquaredSumVector = _mm256_add_epi32(candidateSquaredSumVector, _mm256_madd_epi16(candidateValues, candidateValues)); // ΣC²에 더함
                productSumVector = _mm256_add_epi32(productSumVector, _mm256_madd_epi16(templateValues, candidateValues)); // ΣT×C에 더함
                x += 16;
            }

            // ③ 16개 미만은 행 끝 16개를 읽고 이미 더한 앞부분은 0으로 지움 ────────────────────────────────
            if (x < roiWidth)
            {
                int lastStart = roiWidth - 16; // 행 끝 16개의 시작 위치
                __m128i templateBytes = _mm_loadu_si128(reinterpret_cast<__m128i*>(templateRow + lastStart)); // 템플릿 픽셀 16개 읽기
                __m128i candidateBytes = _mm_loadu_si128(reinterpret_cast<__m128i*>(candidateRow + lastStart)); // 후보 픽셀 16개 읽기
                candidateBytes = _mm_and_si128(candidateBytes, lastBlockMask); // 이미 더한 앞부분을 0으로 (후보가 0이면 T×C도 0)
                __m256i templateValues = _mm256_cvtepu8_epi16(templateBytes); // 8비트 → 16비트
                __m256i candidateValues = _mm256_cvtepu8_epi16(candidateBytes);

                __m256i candidateSumPart = _mm256_sad_epu8(_mm256_zextsi128_si256(candidateBytes), zero); // C를 8개씩 더함
                candidateSumVector = _mm256_add_epi64(candidateSumVector, candidateSumPart); // ΣC에 더함
                candidateSquaredSumVector = _mm256_add_epi32(candidateSquaredSumVector, _mm256_madd_epi16(candidateValues, candidateValues)); // ΣC²에 더함
                productSumVector = _mm256_add_epi32(productSumVector, _mm256_madd_epi16(templateValues, candidateValues)); // ΣT×C에 더함
            }

            addedBlockCount += blockCountPerRow;
            if (addedBlockCount >= 8192) // 32비트 칸이 넘치기 전에 64비트 합으로 옮김
            {
                candidateSquaredSum += SumVectorValues(candidateSquaredSumVector);
                productSum += SumVectorValues(productSumVector);
                candidateSquaredSumVector = zero;
                productSumVector = zero;
                addedBlockCount = 0;
            }
        }

        // 벡터 칸들을 하나의 합으로 모음
        long long candidateSumParts[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(candidateSumParts), candidateSumVector);
        candidateSum = candidateSumParts[0] + candidateSumParts[1] + candidateSumParts[2] + candidateSumParts[3]; // ΣC
        candidateSquaredSum += SumVectorValues(candidateSquaredSumVector); // ΣC²
        productSum += SumVectorValues(productSumVector); // ΣT×C
    }

    // 이미지의 2×2 칸 평균으로 1/2 다운샘플링
    void Downsample(unsigned char* input, int inputWidth, int outputWidth, int outputHeight, unsigned char* output)
    {
        for (int y = 0; y < outputHeight; ++y)
        {
            unsigned char* upperRow = input + (y * 2) * inputWidth; // 2×2 칸의 윗 행
            unsigned char* lowerRow = upperRow + inputWidth; // 2×2 칸의 아랫 행

            for (int x = 0; x < outputWidth; ++x)
            {
                int sum = upperRow[x * 2] + upperRow[x * 2 + 1] + lowerRow[x * 2] + lowerRow[x * 2 + 1];
                output[y * outputWidth + x] = static_cast<unsigned char>((sum + 2) / 4); // 반올림한 평균
            }
        }
    }

    // 선택한 채널의 합을 모두 더해 탐색 범위의 후보마다 점수를 계산해 scores에 저장
    void ComputeCandidateScores(unsigned char* templates[3], unsigned char* results[3], int resultWidth, int roiWidth, int roiHeight, const bool isChannelSelected[3], const std::string& method, int searchLeft, int searchTop, int searchRight, int searchBottom, double* scores)
    {
        bool isDiff = method == "DIFF";
        bool isCoeff = method == "COEFF";
        double pixelCount = static_cast<double>(roiWidth) * roiHeight; // 채널 하나의 ROI 픽셀 수
        long long templateSums[3] = {}; // 채널별 ΣT
        long long templateSquaredTotal = 0; // 선택 채널 전체의 ΣT² (DIFF & CORR 분모)
        double templateVarianceTotal = 0; // 선택 채널 전체의 Σ(T-평균T)² (COEFF 분모)

        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }

            long long templateSquaredSum = 0; // 이 채널의 ΣT²
            for (int y = 0; y < roiHeight; ++y)
            {
                unsigned char* row = templates[channel] + y * roiWidth;

                for (int x = 0; x < roiWidth; ++x)
                {
                    int value = row[x];

                    templateSums[channel] += value; // ΣT
                    templateSquaredSum += value * value; // ΣT²
                }
            }
            templateSquaredTotal += templateSquaredSum;

            double templateVariance = static_cast<double>(templateSquaredSum) - static_cast<double>(templateSums[channel]) * templateSums[channel] / pixelCount; // 이 채널의 Σ(T-평균T)²
            if (templateVariance < 0) { templateVariance = 0; }
            templateVarianceTotal += templateVariance;
        }

        // 탐색 범위의 후보 위치를 탐색
        #pragma omp parallel for schedule(dynamic) // 후보 행들을 여러 코어가 나눠 계산
        for (int y = searchTop; y < searchBottom; ++y)
        {
            for (int x = searchLeft; x < searchRight; ++x)
            {
                long long candidateSquaredTotal = 0; // 선택 채널 전체의 ΣC² (DIFF & CORR용)
                long long productTotal = 0; // 선택 채널 전체의 ΣT×C (DIFF & CORR용)
                double coeffNumeratorTotal = 0; // 선택 채널 전체의 Σ(T-평균T)(C-평균C) (COEFF 분자)
                double candidateVarianceTotal = 0; // 선택 채널 전체의 Σ(C-평균C)² (COEFF 분모)

                for (int channel = 0; channel < 3; ++channel)
                {
                    if (!isChannelSelected[channel]) { continue; }
                    unsigned char* candidateStart = results[channel] + y * resultWidth + x; // 이 채널의 후보 영역 좌상단 주소
                    long long candidateSum; // 이 채널 후보 영역의 ΣC
                    long long candidateSquaredSum; // 이 채널 후보 영역의 ΣC²
                    long long productSum; // 이 채널 후보 영역의 ΣT×C
                    ComputeSumsAvx2(templates[channel], candidateStart, resultWidth, roiWidth, roiHeight, candidateSum, candidateSquaredSum, productSum); // 위 세 변수에 합을 채움

                    if (isCoeff) // COEFF는 채널마다 자기 평균을 빼고 더함 (채널별 밝기 차이에 영향받지 않도록)
                    {
                        coeffNumeratorTotal += static_cast<double>(productSum) - static_cast<double>(templateSums[channel]) * candidateSum / pixelCount; // Σ(T-평균T)(C-평균C)
                        double candidateVariance = static_cast<double>(candidateSquaredSum) - static_cast<double>(candidateSum) * candidateSum / pixelCount; // Σ(C-평균C)²
                        if (candidateVariance < 0) { candidateVariance = 0; } // 계산 오차로 생긴 음수는 0으로
                        candidateVarianceTotal += candidateVariance;
                    }
                    else // DIFF·CORR은 합을 그대로 더함
                    {
                        candidateSquaredTotal += candidateSquaredSum;
                        productTotal += productSum;
                    }
                }

                double score = 0; // 분모가 0이면 0점 (불일치)
                if (isDiff) // DIFF 점수 = 1 - Σ(T-C)² / √(ΣT² × ΣC²), 1에서 빼서 모든 방식이 높을수록 비슷함
                {
                    double denominator = std::sqrt(static_cast<double>(templateSquaredTotal) * candidateSquaredTotal); // 분모 √(ΣT² × ΣC²)
                    long long differenceSquaredTotal = templateSquaredTotal - 2 * productTotal + candidateSquaredTotal; // Σ(T-C)² = ΣT² - 2ΣT×C + ΣC²
                    if (denominator > 0) { score = 1.0 - differenceSquaredTotal / denominator; }
                    if (score < 0) { score = 0; } // 최소 0으로 제한
                }
                else if (isCoeff) // COEFF 점수 = Σ(T-평균T)(C-평균C) / √(Σ(T-평균T)² × Σ(C-평균C)²)
                {
                    double denominator = std::sqrt(templateVarianceTotal * candidateVarianceTotal); // 분모 √(템플릿 분산 × 후보 분산)
                    if (denominator > 0) { score = coeffNumeratorTotal / denominator; }
                    if (score > 1.0) { score = 1.0; } // -1 ~ 1 범위로 제한
                    if (score < -1.0) { score = -1.0; }
                }
                else // CORR 점수 = ΣT×C / √(ΣT² × ΣC²)
                {
                    double denominator = std::sqrt(static_cast<double>(templateSquaredTotal) * candidateSquaredTotal); // 분모 √(ΣT² × ΣC²)
                    if (denominator > 0) { score = static_cast<double>(productTotal) / denominator; }
                    if (score > 1.0) { score = 1.0; } // 최대 1로 제한
                }

                scores[(y - searchTop) * (searchRight - searchLeft) + (x - searchLeft)] = score; // 탐색 범위 안에서의 위치에 저장
            }
        }
    }
}

#pragma managed(pop)

namespace
{
    const double DIFF_MAXIMUM_SCORE = 0.1; // DIFF 점수가 이 값 이하면 매칭
    const double CORR_MINIMUM_SCORE = 0.95; // CORR 점수가 이 값 이상이면 매칭
    const double COEFF_MINIMUM_SCORE = 0.72; // COEFF 점수가 이 값 이상이면 매칭
    const double DIFF_COARSE_MAXIMUM_SCORE = 0.3; // 줄인 단계에서 DIFF 점수가 이 값 이하면 다음 단계로 넘김
    const double CORR_COARSE_MINIMUM_SCORE = 0.85; // 줄인 단계에서 CORR 점수가 이 값 이상이면 다음 단계로 넘김
    const double COEFF_COARSE_MINIMUM_SCORE = 0.5; // 줄인 단계에서 COEFF 점수가 이 값 이상이면 다음 단계로 넘김
    const long long STORED_SCORE_SCALE = 1000000; // 점수를 정수로 저장할 배율
    
    array<array<Int64>^>^ KeepBestMatches(List<int>^ matchPositions, List<double>^ matchScores, int candidateCountX, int candidateCountY, int roiWidth, int roiHeight)
    {
        // 후보를 점수가 높은 순서로 정렬 (점수가 같으면 앞 위치 먼저)
        int matchCount = matchPositions->Count; // 기준을 통과한 후보 수
        array<int>^ sortedPositions = matchPositions->ToArray(); // 점수순으로 정렬할 후보 위치 목록
        array<double>^ negatedScores = gcnew array<double>(matchCount);
        for (int i = 0; i < matchCount; ++i) { negatedScores[i] = -matchScores[i]; } // 큰 점수가 앞에 오도록 부호를 뒤집음
        Array::Sort(negatedScores, sortedPositions); // 점수 순서대로 위치도 함께 정렬

        int sameScoreStart = 0;
        while (sameScoreStart < matchCount) // 점수가 같은 구간은 위치 순서로 다시 정렬
        {
            int sameScoreEnd = sameScoreStart + 1;
            while (sameScoreEnd < matchCount && negatedScores[sameScoreEnd] == negatedScores[sameScoreStart]) { sameScoreEnd++; }
            if (sameScoreEnd - sameScoreStart > 1) { Array::Sort(sortedPositions, sameScoreStart, sameScoreEnd - sameScoreStart); }
            sameScoreStart = sameScoreEnd;
        }

        // ROI 크기의 칸으로 나눈 격자에 고른 후보를 기록 (겹치지 않으므로 한 칸에 최대 1개)
        int cellCountX = candidateCountX / roiWidth + 1; // 가로 칸 수
        int cellCountY = candidateCountY / roiHeight + 1; // 세로 칸 수
        array<int>^ acceptedInCell = gcnew array<int>(cellCountX * cellCountY); // 칸마다 채택된 후보 위치
        for (int i = 0; i < acceptedInCell->Length; ++i) { acceptedInCell[i] = -1; } // -1은 빈 칸

        List<array<Int64>^>^ finalMatches = gcnew List<array<Int64>^>(); // 최종 후보 목록
        for (int i = 0; i < matchCount; ++i) // 점수가 높은 후보부터 차례로
        {
            int candidatePosition = sortedPositions[i];
            int x = candidatePosition % candidateCountX; // 위치 번호를 x, y 좌표로 변환
            int y = candidatePosition / candidateCountX;
            int cellX = x / roiWidth; // 이 후보가 속한 칸
            int cellY = y / roiHeight;

            // 주변 9칸의 채택된 후보와 겹치는지 확인
            bool isOverlapping = false;
            for (int nearY = cellY - 1; nearY <= cellY + 1; ++nearY)
            {
                for (int nearX = cellX - 1; nearX <= cellX + 1; ++nearX)
                {
                    if (nearX < 0 || nearY < 0 || nearX >= cellCountX || nearY >= cellCountY) { continue; } // 격자 밖은 건너뜀
                    int acceptedPosition = acceptedInCell[nearY * cellCountX + nearX];
                    if (acceptedPosition < 0) { continue; } // 빈 칸은 건너뜀

                    int acceptedX = acceptedPosition % candidateCountX;
                    int acceptedY = acceptedPosition / candidateCountX;
                    if (Math::Abs(x - acceptedX) < roiWidth && Math::Abs(y - acceptedY) < roiHeight) { isOverlapping = true; } // 가로·세로 모두 가까우면 겹침
                }
            }
            if (isOverlapping) { continue; } // 겹치면 버림

            acceptedInCell[cellY * cellCountX + cellX] = candidatePosition; // 채택된 후보를 칸에 기록
            Int64 storedScore = static_cast<Int64>(Math::Round(-negatedScores[i] * STORED_SCORE_SCALE)); // 정렬된 점수(부호를 되돌림)를 정수로 변환
            finalMatches->Add(gcnew array<Int64> { x, y, roiWidth, roiHeight, storedScore, STORED_SCORE_SCALE }); // [x, y, 폭, 높이, 점수, 최대 점수]
        }

        return finalMatches->ToArray();
    }
    array<array<Int64>^>^ MatchTemplate(int sourceWidth, int sourceHeight, array<Byte>^ sourceRedPixels, array<Byte>^ sourceGreenPixels, array<Byte>^ sourceBluePixels, int resultWidth, int resultHeight, array<Byte>^ resultRedPixels, array<Byte>^ resultGreenPixels, array<Byte>^ resultBluePixels, int roiX, int roiY, int roiWidth, int roiHeight, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, std::string method)
    {
        // 원본 단계의 최종 기준과 줄인 단계의 기준 (DIFF는 1 - D로 뒤집어 모두 높을수록 비슷하다는 의미)
        double minimumScore = CORR_MINIMUM_SCORE;
        double coarseMinimumScore = CORR_COARSE_MINIMUM_SCORE; // 줄이면 참 위치 점수도 떨어지므로 더 느슨하게
        if (method == "DIFF") { minimumScore = 1.0 - DIFF_MAXIMUM_SCORE; coarseMinimumScore = 1.0 - DIFF_COARSE_MAXIMUM_SCORE; }
        else if (method == "COEFF") { minimumScore = COEFF_MINIMUM_SCORE; coarseMinimumScore = COEFF_COARSE_MINIMUM_SCORE; }

        // 네이티브 함수에 넘기도록 배열 주소 고정 (선택하지 않은 채널은 고정만 하고 읽지 않음)
        pin_ptr<Byte> sourceRedPointer = &sourceRedPixels[0];
        pin_ptr<Byte> sourceGreenPointer = &sourceGreenPixels[0];
        pin_ptr<Byte> sourceBluePointer = &sourceBluePixels[0];
        pin_ptr<Byte> resultRedPointer = &resultRedPixels[0];
        pin_ptr<Byte> resultGreenPointer = &resultGreenPixels[0];
        pin_ptr<Byte> resultBluePointer = &resultBluePixels[0];

        unsigned char* sources[3] = { sourceRedPointer, sourceGreenPointer, sourceBluePointer }; // 템플릿을 잘라올 이미지 R, G, B
        unsigned char* results[3] = { resultRedPointer, resultGreenPointer, resultBluePointer }; // 결과 R, G, B
        bool isChannelSelected[3] = { isRedSelected, isGreenSelected, isBlueSelected }; // 채널별 선택 여부

        // 피라미드 단계 수: 줄인 ROI가 16픽셀 이상인 동안 가로·세로 1/2씩 계속 줄임 (0단계 = 원본, 1 << n = n단계의 배율 2ⁿ)
        int levelCount = 1;
        while (roiWidth / (1 << levelCount) >= 16 && roiHeight / (1 << levelCount) >= 16) { levelCount++; } // 16 미만이면 세부가 사라지고 AVX2 경로(폭 16 이상)도 못 씀

        // 단계별 R, G, B 이미지
        std::vector<std::vector<unsigned char>> templateBuffers(levelCount * 3); // 단계별 템플릿 (ROI만 담음)
        std::vector<std::vector<unsigned char>> resultBuffers(levelCount * 3); // 단계별 결과 이미지 (0단계는 원본 배열을 그대로 쓰므로 비어 있음)
        std::vector<unsigned char*> templatePointers(levelCount * 3); // 단계별 템플릿 주소
        std::vector<unsigned char*> resultPointers(levelCount * 3); // 단계별 결과 이미지 주소

        // STEP 0: 템플릿은 source 이미지의 ROI를 복사하고, 결과 이미지는 넘겨받은 배열을 그대로 가리킴
        for (int channel = 0; channel < 3; ++channel)
        {
            if (!isChannelSelected[channel]) { continue; }

            templateBuffers[channel].resize(roiWidth * roiHeight);
            for (int y = 0; y < roiHeight; ++y) { std::memcpy(templateBuffers[channel].data() + y * roiWidth, sources[channel] + (roiY + y) * sourceWidth + roiX, roiWidth); } // ROI 한 행씩 복사
            templatePointers[channel] = templateBuffers[channel].data();
            resultPointers[channel] = results[channel];
        }

        // 1단계부터는 바로 아래 단계를 2×2 평균으로 줄여 만듦
        for (int level = 1; level < levelCount; ++level)
        {
            int previousScale = 1 << (level - 1); // 바로 아래 단계의 배율
            int scale = 1 << level; // 이 단계의 배율

            for (int channel = 0; channel < 3; ++channel)
            {
                if (!isChannelSelected[channel]) { continue; }
                int previousIndex = (level - 1) * 3 + channel; // 바로 아래 단계의 칸 번호
                int currentIndex = level * 3 + channel; // 이 단계의 칸 번호

                templateBuffers[currentIndex].resize((roiWidth / scale) * (roiHeight / scale));
                resultBuffers[currentIndex].resize((resultWidth / scale) * (resultHeight / scale));
                Downsample(templatePointers[previousIndex], roiWidth / previousScale, roiWidth / scale, roiHeight / scale, templateBuffers[currentIndex].data());
                Downsample(resultPointers[previousIndex], resultWidth / previousScale, resultWidth / scale, resultHeight / scale, resultBuffers[currentIndex].data());
                templatePointers[currentIndex] = templateBuffers[currentIndex].data();
                resultPointers[currentIndex] = resultBuffers[currentIndex].data();
            }
        }

        // STEP 1: 가장 작은 단계에서 모든 후보 위치를 탐색
        int topLevel = levelCount - 1;
        int topScale = 1 << topLevel;
        int topRoiWidth = roiWidth / topScale;
        int topRoiHeight = roiHeight / topScale;
        int topResultWidth = resultWidth / topScale;
        int topCandidateCountX = topResultWidth - topRoiWidth + 1; // 가장 작은 단계의 가로 후보 위치 수
        int topCandidateCountY = resultHeight / topScale - topRoiHeight + 1; // 가장 작은 단계의 세로 후보 위치 수

        std::vector<double> topScores(topCandidateCountX * topCandidateCountY); // 가장 작은 단계의 후보 위치별 점수
        ComputeCandidateScores(&templatePointers[topLevel * 3], &resultPointers[topLevel * 3], topResultWidth, topRoiWidth, topRoiHeight, isChannelSelected, method, 0, 0, topCandidateCountX, topCandidateCountY, topScores.data());

        if (levelCount == 1) // 줄이지 않았으면 기준을 통과한 위치가 최종 후보
        {
            List<int>^ matchPositions = gcnew List<int>(); // 기준을 통과한 위치
            List<double>^ matchScores = gcnew List<double>(); // 그 위치의 점수
            for (int position = 0; position < static_cast<int>(topScores.size()); ++position)
            {
                if (topScores[position] >= minimumScore) { matchPositions->Add(position); matchScores->Add(topScores[position]); }
            }

            return KeepBestMatches(matchPositions, matchScores, topCandidateCountX, topCandidateCountY, roiWidth, roiHeight); // 겹치지 않는 최고 점수 후보만 반환
        }

        // 줄였으면 줄인 단계 기준을 통과하고 주변 8칸보다 낮지 않은 위치를 모두 남김 (반복 무늬에서 참 위치가 지워지지 않도록 여기서는 겹침 제거를 하지 않음)
        std::vector<int> peakXs; // 봉우리 x 좌표 (현재 단계 기준)
        std::vector<int> peakYs; // 봉우리 y 좌표 (현재 단계 기준)
        for (int position = 0; position < static_cast<int>(topScores.size()); ++position)
        {
            double score = topScores[position];
            if (score < coarseMinimumScore) { continue; }

            int x = position % topCandidateCountX;
            int y = position / topCandidateCountX;
            bool isPeak = true;
            for (int nearY = y - 1; nearY <= y + 1; ++nearY)
            {
                for (int nearX = x - 1; nearX <= x + 1; ++nearX)
                {
                    if (nearX < 0 || nearY < 0 || nearX >= topCandidateCountX || nearY >= topCandidateCountY) { continue; } // 범위 밖은 건너뜀
                    int nearPosition = nearY * topCandidateCountX + nearX;
                    double nearScore = topScores[nearPosition];
                    if (nearScore > score || (nearScore == score && nearPosition < position)) { isPeak = false; } // 더 높거나, 같은데 앞에 있으면 봉우리가 아님
                }
            }

            if (isPeak) { peakXs.push_back(x); peakYs.push_back(y); }
        }

        // STEP 2: 한 단계씩 내려가며 봉우리 주변 ±3칸만 다시 계산해 가장 좋은 위치로 옮김
        int peakCount = static_cast<int>(peakXs.size());
        std::vector<double> peakScores(peakCount); // 봉우리 점수

        for (int level = topLevel - 1; level >= 0; --level)
        {
            int scale = 1 << level; // 이 단계의 배율
            int levelRoiWidth = roiWidth / scale;
            int levelRoiHeight = roiHeight / scale;
            int levelResultWidth = resultWidth / scale;
            int levelCandidateCountX = levelResultWidth - levelRoiWidth + 1; // 이 단계의 가로 후보 위치 수
            int levelCandidateCountY = resultHeight / scale - levelRoiHeight + 1; // 이 단계의 세로 후보 위치 수

            #pragma omp parallel for schedule(dynamic) // 봉우리들을 여러 코어가 나눠 계산
            for (int i = 0; i < peakCount; ++i)
            {
                int searchLeft = Math::Max(0, peakXs[i] * 2 - 3); // 한 단계 위 좌표 × 2 주변 ±3칸 (2칸이면 반 픽셀 어긋남과 1칸 오차가 겹칠 때 못 덮음)
                int searchTop = Math::Max(0, peakYs[i] * 2 - 3);
                int searchRight = Math::Min(levelCandidateCountX, peakXs[i] * 2 + 4); // 오른쪽·아래 경계는 포함하지 않으므로 +3 대신 +4
                int searchBottom = Math::Min(levelCandidateCountY, peakYs[i] * 2 + 4);
                int searchWidth = searchRight - searchLeft;

                std::vector<double> windowScores(searchWidth * (searchBottom - searchTop)); // 창 안의 후보 위치별 점수 (최대 7×7칸)
                ComputeCandidateScores(&templatePointers[level * 3], &resultPointers[level * 3], levelResultWidth, levelRoiWidth, levelRoiHeight, isChannelSelected, method, searchLeft, searchTop, searchRight, searchBottom, windowScores.data());

                peakScores[i] = -2.0; // 모든 점수(-1 ~ 1)보다 낮은 값에서 시작해 창 안의 최고 점수를 찾음
                for (int j = 0; j < static_cast<int>(windowScores.size()); ++j)
                {
                    if (windowScores[j] > peakScores[i])
                    {
                        peakScores[i] = windowScores[j];
                        peakXs[i] = searchLeft + j % searchWidth; // 창 안의 위치를 이 단계 좌표로 변환
                        peakYs[i] = searchTop + j / searchWidth;
                    }
                }
            }

            // 기준을 못 넘었거나 앞 봉우리와 같은 위치로 모인 봉우리는 버림 (다음 단계 계산을 줄임)
            double levelMinimumScore = coarseMinimumScore; // 줄인 단계는 느슨한 기준
            if (level == 0) { levelMinimumScore = minimumScore; } // 원본 단계는 최종 기준
            int keptCount = 0; // 남긴 봉우리 수
            for (int i = 0; i < peakCount; ++i)
            {
                if (peakScores[i] < levelMinimumScore) { continue; } // 가망 없는 봉우리

                bool isDuplicate = false;
                for (int j = 0; j < keptCount; ++j)
                {
                    if (peakXs[j] == peakXs[i] && peakYs[j] == peakYs[i]) { isDuplicate = true; } // 이미 남긴 봉우리와 같은 위치
                }
                if (isDuplicate) { continue; }

                peakXs[keptCount] = peakXs[i]; // 남길 봉우리를 목록 앞쪽으로 당겨 채움
                peakYs[keptCount] = peakYs[i];
                peakScores[keptCount] = peakScores[i];
                keptCount++;
            }
            peakCount = keptCount;
        }

        // STEP 3: 원본 단계까지 남은 봉우리를 모아 겹침 제거
        int candidateCountX = resultWidth - roiWidth + 1; // 가로 후보 위치 수
        int candidateCountY = resultHeight - roiHeight + 1; // 세로 후보 위치 수
        List<int>^ matchPositions = gcnew List<int>();
        List<double>^ matchScores = gcnew List<double>();
        for (int i = 0; i < peakCount; ++i) { matchPositions->Add(peakYs[i] * candidateCountX + peakXs[i]); matchScores->Add(peakScores[i]); }

        return KeepBestMatches(matchPositions, matchScores, candidateCountX, candidateCountY, roiWidth, roiHeight); // 겹치지 않는 최고 점수 후보만 반환
    }
}

namespace Vision::CppCli
{
    array<array<Int64>^>^ TemplateMatching::MatchDiff(ColorImage^ source, ColorImage^ result, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected)
    {
        return MatchTemplate(source->Width, source->Height, source->RedPixels, source->GreenPixels, source->BluePixels, result->Width, result->Height, result->RedPixels, result->GreenPixels, result->BluePixels, roi->X, roi->Y, roi->Width, roi->Height, isRedSelected, isGreenSelected, isBlueSelected, "DIFF");
    }
    array<array<Int64>^>^ TemplateMatching::MatchCorr(ColorImage^ source, ColorImage^ result, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected)
    {
        return MatchTemplate(source->Width, source->Height, source->RedPixels, source->GreenPixels, source->BluePixels, result->Width, result->Height, result->RedPixels, result->GreenPixels, result->BluePixels, roi->X, roi->Y, roi->Width, roi->Height, isRedSelected, isGreenSelected, isBlueSelected, "CORR");
    }
    array<array<Int64>^>^ TemplateMatching::MatchCoeff(ColorImage^ source, ColorImage^ result, Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected)
    {
        return MatchTemplate(source->Width, source->Height, source->RedPixels, source->GreenPixels, source->BluePixels, result->Width, result->Height, result->RedPixels, result->GreenPixels, result->BluePixels, roi->X, roi->Y, roi->Width, roi->Height, isRedSelected, isGreenSelected, isBlueSelected, "COEFF");
    }
}
