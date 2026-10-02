using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Vision.Core;

namespace Vision.Ui
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private BmpFileReader bmpFileReader; // BMP 파일 읽기 객체
        public MainViewModel(BmpFileReader bmpFileReader) { this.bmpFileReader = bmpFileReader; }
        public event PropertyChangedEventHandler? PropertyChanged;
        private void NotifyPropertiesChanged(params string[] propertyNames)
        {
            foreach (string propertyName in propertyNames)
            {
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        // ── 이미지와 표시 모드 상태값 ───────────────────────────────────────────────
        private ColorImage? sourceImage; // 좌측 원본 이미지
        private ColorImage? resultImage; // 우측 결과(작업) 이미지
        private string selectedImageViewMode = "FullColor"; // 이미지 표시 모드 (FullColor/Red/Green/Blue)
        public string SelectedImageViewMode // 현재 선택된 이미지 표시 모드
        {
            get { return selectedImageViewMode; } // 저장된 표시 모드를 반환
            set // 새로운 표시 모드를 설정할 때 실행
            {
                selectedImageViewMode = value; // 새 표시 모드를 저장
                NotifyPropertiesChanged(nameof(SelectedImageViewMode)); // 화면에 변경 내용을 알림
            }
        }


        // ── 내부 작업 상태값 ────────────────────────────────────────────────────
        private bool isBusy; // 영상처리 진행중 여부
        public bool IsBusy
        {
            get { return isBusy; }
            private set
            {
                isBusy = value;
                NotifyPropertiesChanged(nameof(IsBusy), nameof(IsIdle), nameof(StatusBadgeText));
            }
        }
        public bool IsIdle { get { return !IsBusy; } }
        private string statusMessage = "BMP 파일을 열어 주세요."; // 하단 상태바 메세지
        public string StatusMessage
        {
            get { return statusMessage; }
            private set
            {
                statusMessage = value;
                NotifyPropertiesChanged(nameof(StatusMessage));
            }
        }
        public string StatusBadgeText { get { return IsBusy ? "대기" : "완료"; } }


        // ── 이미지 값 ────────────────────────────────────────────────────────
        public ColorImage? SourceImage // 좌측 원본 이미지. 파일을 열 때만 갱신
        {
            get { return sourceImage; }
            private set
            {
                sourceImage = value;
                NotifyPropertiesChanged(nameof(SourceImage), nameof(HasSourceImage), nameof(CanRun));
            }
        }
        public ColorImage? ResultImage // 우측 결과 이미지. (다음 처리의 입력으로도 사용, 처리 함수가 이 이미지의 배열을 직접 수정)
        {
            get { return resultImage; }
            private set
            {
                resultImage = value;
                NotifyPropertiesChanged(nameof(ResultImage), nameof(HasResultImage));
            }
        }
        public bool HasSourceImage { get { return SourceImage != null; } } // 원본 이미지가 불러와졌는지 여부
        public bool HasResultImage { get { return ResultImage != null; } } // 결과 이미지가 존재하는지 여부(저장 버튼 활성화 조건)
        public bool CanRun { get { return HasSourceImage && HasSelectedChannel; } } // 원본과 처리 대상 채널이 모두 준비되었는지 여부


        // ── 히스토그램 값 ──────────────────────────────────────────────────────
        public double[] RedHistogram { get; private set; } = new double[256];
        public double[] GreenHistogram { get; private set; } = new double[256];
        public double[] BlueHistogram { get; private set; } = new double[256];
        private void UpdateHistogram() // 현재 ResultImage와 ROI 설정에 맞춰 히스토그램 다시 계산
        {
            ColorImage image = ResultImage!;

            Roi roi = CurrentRoi;
            HistogramData histogramData = Vision.CppCli.Histogram.Calculate(image, roi);

            RedHistogram = histogramData.NormalizedRed;
            GreenHistogram = histogramData.NormalizedGreen;
            BlueHistogram = histogramData.NormalizedBlue;

            NotifyPropertiesChanged(nameof(RedHistogram), nameof(GreenHistogram), nameof(BlueHistogram));
        }


        // ── ROI 값 ──────────────────────────────────────────────────────────
        private Roi currentRoi = Roi.Empty; // 현재 선택된 ROI 영역
        public Roi CurrentRoi
        {
            get { return currentRoi; }
            private set
            {
                currentRoi = value;
                NotifyPropertiesChanged(nameof(CurrentRoi), nameof(IsRoiSelected));
            }
        }
        public bool IsRoiSelected { get { return !CurrentRoi.IsEmpty; } }
        public void SetRoi(Roi roi) // UI에서 생성한 ROI 영역으로 히스토그램을 갱신
        {
            CurrentRoi = roi;
            UpdateHistogram();
            ClearTemplateMatches();
        }
        public void ClearRoi() // ROI 해제하고 전체 이미지로 히스토그램 업데이트
        {
            CurrentRoi = Roi.Empty;
            UpdateHistogram();
            ClearTemplateMatches();
        }


        // ── 템플릿 매칭 값 ──────────────────────────────────────────────────────
        public long[][] TemplateMatches { get; private set; } = Array.Empty<long[]>();
        private void ClearTemplateMatches() // 새 이미지, ROI 변경, 일반 영상처리 뒤 오래된 매칭 결과를 비움
        {
            TemplateMatches = Array.Empty<long[]>();
            NotifyPropertiesChanged(nameof(TemplateMatches));
        }


        // ── [대상 채널] 버튼 선택값 ─────────────────────────────────────────────────
        private bool isRedChannelSelected = true; // Red 채널 선택 여부
        private bool isGreenChannelSelected = true; // Green 채널 선택 여부
        private bool isBlueChannelSelected = true; // Blue 채널 선택 여부
        public bool IsRedChannelSelected
        {
            get { return isRedChannelSelected; }
            set
            {
                isRedChannelSelected = value;
                NotifyPropertiesChanged(nameof(IsRedChannelSelected), nameof(HasSelectedChannel), nameof(CanRun));
            }
        }
        public bool IsGreenChannelSelected
        {
            get { return isGreenChannelSelected; }
            set
            {
                isGreenChannelSelected = value;
                NotifyPropertiesChanged(nameof(IsGreenChannelSelected), nameof(HasSelectedChannel), nameof(CanRun));
            }
        }
        public bool IsBlueChannelSelected
        {
            get { return isBlueChannelSelected; }
            set
            {
                isBlueChannelSelected = value;
                NotifyPropertiesChanged(nameof(IsBlueChannelSelected), nameof(HasSelectedChannel), nameof(CanRun));
            }
        }
        public bool HasSelectedChannel { get { return IsRedChannelSelected || IsGreenChannelSelected || IsBlueChannelSelected; } }


        // ── [커널 설정] 설정값 ───────────────────────────────────────────────────
        private int kernelSize = 3; // 공간 필터링 커널 크기
        public int KernelSize
        {
            get { return kernelSize; }
            set
            {
                int clamped = Math.Max(3, value);
                if (clamped % 2 == 0) { clamped += 1; } // 커널 크기는 항상 홀수여야 함
                kernelSize = clamped;
                NotifyPropertiesChanged(nameof(KernelSize));
            }
        }
        private double sigma = 1.0; // 가우시안 시그마 값
        public double Sigma
        {
            get { return sigma; }
            set
            {
                sigma = value;
                NotifyPropertiesChanged(nameof(Sigma));
            }
        }


        // ── 영상처리 실행 메서드 ──────────────────────────────────────────────────
        public async Task OpenImageAsync(string path) // 비동기로 BMP 이미지 파일 읽고 뷰 상태 초기화
        {
            IsBusy = true;
            StatusMessage = "이미지를 여는 중...";

            try
            {
                ColorImage image = await Task.Run(() => bmpFileReader.ReadColorBmp(path));

                SelectedImageViewMode = "FullColor";
                CurrentRoi = Roi.Empty;
                SourceImage = image; // 좌측 원본 뷰 갱신
                ResultImage = image.Clone(); // 좌측 원본이 바뀌지 않도록 복제본으로 우측 결과 뷰·내비게이터 갱신
                UpdateHistogram();
                ClearTemplateMatches();

                StatusMessage = $"{Path.GetFileName(path)} ({image.Width} × {image.Height})";
            }
            finally
            {
                IsBusy = false;
            }
        }
        public async Task ResetImageAsync() // 좌측 원본을 다시 복제해 우측 결과 이미지를 초기화
        {
            ColorImage? source = SourceImage;
            if (source == null) { return; }

            IsBusy = true;
            StatusMessage = "원본 이미지로 초기화하는 중...";

            try
            {
                ColorImage image = await Task.Run(() => source.Clone()); // 원본이 바뀌지 않도록 복제본 생성

                SelectedImageViewMode = "FullColor"; // 이미지를 처음 열었을 때와 같은 상태로 되돌림
                CurrentRoi = Roi.Empty;
                ResultImage = image;
                UpdateHistogram();
                ClearTemplateMatches();

                StatusMessage = "원본 이미지로 초기화 완료";
            }
            finally
            {
                IsBusy = false;
            }
        }
        public async Task RunGaussianAsync() { await RunAlgorithmAsync("가우시안", (image, roi, isRedSelected, isGreenSelected, isBlueSelected) => Vision.CppCli.SpatialFilter.Gaussian(image, roi, isRedSelected, isGreenSelected, isBlueSelected, KernelSize, Sigma)); }
        public async Task RunLaplacianAsync() { await RunAlgorithmAsync("라플라시안", (image, roi, isRedSelected, isGreenSelected, isBlueSelected) => Vision.CppCli.SpatialFilter.Laplacian(image, roi, isRedSelected, isGreenSelected, isBlueSelected, KernelSize, Sigma)); }
        public async Task RunSobelAsync() { await RunAlgorithmAsync("소벨", (image, roi, isRedSelected, isGreenSelected, isBlueSelected) => Vision.CppCli.SpatialFilter.Sobel(image, roi, isRedSelected, isGreenSelected, isBlueSelected, KernelSize)); }
        public async Task RunBinarizationAsync() { await RunAlgorithmAsync("이진화", Vision.CppCli.Histogram.Binarize); }
        public async Task RunEqualizationAsync() { await RunAlgorithmAsync("평활화", Vision.CppCli.Histogram.Equalize); }
        public async Task RunDilationAsync() { await RunAlgorithmAsync("팽창", (image, roi, isRedSelected, isGreenSelected, isBlueSelected) => Vision.CppCli.Morphology.Dilate(image, roi, isRedSelected, isGreenSelected, isBlueSelected, KernelSize)); }
        public async Task RunErosionAsync() { await RunAlgorithmAsync("수축", (image, roi, isRedSelected, isGreenSelected, isBlueSelected) => Vision.CppCli.Morphology.Erode(image, roi, isRedSelected, isGreenSelected, isBlueSelected, KernelSize)); }
        public async Task RunDiffAsync() { await RunMatchingAsync("DIFF", Vision.CppCli.TemplateMatching.MatchDiff); }
        public async Task RunCorrAsync() { await RunMatchingAsync("CORR", Vision.CppCli.TemplateMatching.MatchCorr); }
        public async Task RunCoeffAsync() { await RunMatchingAsync("COEFF", Vision.CppCli.TemplateMatching.MatchCoeff); }
        private async Task RunAlgorithmAsync(string operationName, Action<ColorImage, Roi, bool, bool, bool> algorithm)
        {
            ColorImage? input = ResultImage;
            if (input == null) { return; }

            Roi roi = CurrentRoi;
            bool isRedSelected = IsRedChannelSelected;
            bool isGreenSelected = IsGreenChannelSelected;
            bool isBlueSelected = IsBlueChannelSelected;

            IsBusy = true;
            StatusMessage = "알고리즘을 실행하는 중...";

            try
            {
                Stopwatch stopwatch = Stopwatch.StartNew(); // 세 채널 병렬 실행 시간을 측정
                await Task.Run(() => algorithm(input, roi, isRedSelected, isGreenSelected, isBlueSelected)); // 선택 채널은 C++/CLI 안에서 병렬 처리
                stopwatch.Stop(); // 실행 시간 측정 종료

                ColorImage output = new ColorImage(input.Width, input.Height, input.RedPixels, input.GreenPixels, input.BluePixels); // 처리 함수가 입력 배열을 직접 수정하므로, 같은 배열로 새 이미지를 만들어 화면 갱신을 알림
                ResultImage = output;
                UpdateHistogram();
                ClearTemplateMatches();

                double elapsedMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
                StatusMessage = $"{operationName} 적용 완료 · {elapsedMilliseconds:F2} ms";
            }
            finally
            {
                IsBusy = false;
            }
        }
        private async Task RunMatchingAsync(string operationName, Func<ColorImage, ColorImage, Roi, bool, bool, bool, long[][]> algorithm)
        {
            ColorImage? result = ResultImage;
            if (result == null || !IsRoiSelected || !HasSelectedChannel || IsBusy) { return; }

            Roi roi = CurrentRoi;
            if (roi.Width < 16) { StatusMessage = "템플릿 매칭은 ROI 폭 16픽셀 이상에서만 가능합니다."; return; } // AVX2가 한 행을 16픽셀 단위로 읽음
            bool isRedSelected = IsRedChannelSelected;
            bool isGreenSelected = IsGreenChannelSelected;
            bool isBlueSelected = IsBlueChannelSelected;

            IsBusy = true;
            StatusMessage = "알고리즘을 실행하는 중...";
            Stopwatch stopwatch = Stopwatch.StartNew();

            try
            {
                long[][] matches = await Task.Run(() => algorithm(result, result, roi, isRedSelected, isGreenSelected, isBlueSelected)); // 템플릿도 결과 이미지 ROI에서 잘라 탐색 이미지와 처리 상태를 맞춤
                stopwatch.Stop();

                TemplateMatches = matches;
                NotifyPropertiesChanged(nameof(TemplateMatches));

                StatusMessage = $"{operationName} 템플릿 매칭 완료 · {stopwatch.Elapsed.TotalMilliseconds:F2} ms";
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}
