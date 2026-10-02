using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using Vision.Core;

namespace Vision.Ui
{
    public partial class MainWindow : Window
    {
        private MainViewModel mainViewModel;
        private BitmapSource? sourceFullColorBitmap;
        private BitmapSource? resultFullColorBitmap;
        private int leftClickX = -1;
        private int leftClickY = -1;
        private int rightClickX = -1;
        private int rightClickY = -1;

        public MainWindow()
        {
            InitializeComponent();

            mainViewModel = new MainViewModel(new BmpFileReader());
            DataContext = mainViewModel;
            mainViewModel.PropertyChanged += MainViewModelPropertyChanged;
        }

        // MainViewModel 변경 처리 ─────────────────────────────────────────────────
        private void MainViewModelPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.SourceImage))
            {
                ClearClicks();
                sourceFullColorBitmap = CreateBitmap(mainViewModel.SourceImage, "FullColor");
                UpdateSourceImageView();
            }
            else if (e.PropertyName == nameof(MainViewModel.ResultImage))
            {
                UpdateResultFullColorBitmap();
                UpdateResultImageView();
                UpdateNavigatorImageView();
                UpdateTemplateMatchOverlay();
            }
            else if (e.PropertyName == nameof(MainViewModel.SelectedImageViewMode))
            {
                UpdateSourceImageView();
                UpdateResultImageView();
            }
            else if (e.PropertyName == nameof(MainViewModel.CurrentRoi))
            {
                UpdateRoiOverlay(SourceImageGrid, SourceRoiRectangle);
                UpdateRoiOverlay(ResultImageGrid, ResultRoiRectangle);
                UpdateNavigatorImageView();
                UpdateTemplateMatchOverlay();
            }
            else if (e.PropertyName == nameof(MainViewModel.TemplateMatches))
            {
                UpdateTemplateMatchOverlay();
            }
        }


        // 상단 UI 버튼 클릭 ─────────────────────────────────────────────────
        private async void OpenImageButtonClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog { Filter = "BMP 이미지 (*.bmp)|*.bmp" };

            if (dialog.ShowDialog() != true) { return; }

            await mainViewModel.OpenImageAsync(dialog.FileName);
        }
        private async void ResetImageButtonClick(object sender, RoutedEventArgs e)
        {
            ClearClicks(); // 이전 클릭 좌표가 다음 ROI에 섞이지 않도록 비움

            await mainViewModel.ResetImageAsync();
        }
        private async void GaussianButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunGaussianAsync(); }
        private async void LaplacianButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunLaplacianAsync(); }
        private async void SobelButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunSobelAsync(); }
        private async void BinarizationButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunBinarizationAsync(); }
        private async void EqualizationButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunEqualizationAsync(); }
        private async void DilationButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunDilationAsync(); }
        private async void ErosionButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunErosionAsync(); }
        private async void DiffButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunDiffAsync(); }
        private async void CorrButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunCorrAsync(); }
        private async void CoeffButtonClick(object sender, RoutedEventArgs e) { await mainViewModel.RunCoeffAsync(); }


        // 이미지 표시 ────────────────────────────────────────────────────
        private BitmapSource? CreateBitmap(ColorImage? image, string mode) // 선택한 mode에 맞는 WPF 비트맵 생성
        {
            if (image == null) { return null; }

            if (mode == "FullColor")
            {
                return BitmapSource.Create(image.Width, image.Height, 96, 96, PixelFormats.Bgr24, null, image.ToBgrBytes(), image.Width * 3);
            }
            else
            {

                byte[] pixels;
                if (mode == "Red") { pixels = image.RedPixels; }
                else if (mode == "Green") { pixels = image.GreenPixels; }
                else if (mode == "Blue") { pixels = image.BluePixels; }
                else { return null; }

                return BitmapSource.Create(image.Width, image.Height, 96, 96, PixelFormats.Gray8, null, pixels, image.Width);
            }
        }
        private void UpdateResultFullColorBitmap() // 결과 이미지를 전체 색상 비트맵으로 캐시
        {
            ColorImage? image = mainViewModel.ResultImage;

            if (image == null)
            {
                resultFullColorBitmap = null;
            }
            else
            {
                resultFullColorBitmap = CreateBitmap(image, "FullColor");
            }
        }
        private void UpdateSourceImageView() // 좌측 원본 이미지를 현재 모드로 표시
        {
            if (mainViewModel.SelectedImageViewMode == "FullColor")
            {
                SourceImageView.Source = sourceFullColorBitmap;
            }
            else
            {
                SourceImageView.Source = CreateBitmap(mainViewModel.SourceImage, mainViewModel.SelectedImageViewMode);
            }
        }
        private void UpdateResultImageView() // 우측 결과 이미지를 현재 모드로 표시
        {
            if (mainViewModel.SelectedImageViewMode == "FullColor")
            {
                ResultImageView.Source = resultFullColorBitmap;
            }
            else
            {
                ResultImageView.Source = CreateBitmap(mainViewModel.ResultImage, mainViewModel.SelectedImageViewMode);
            }
        }
        private void UpdateNavigatorImageView() // ROI에 맞춰 내비게이터 이미지를 표시
        {
            Roi roi = mainViewModel.CurrentRoi;

            if (resultFullColorBitmap == null || roi.IsEmpty)
            {
                NavigatorImageView.Source = resultFullColorBitmap;
            }
            else
            {
                NavigatorImageView.Source = new CroppedBitmap(resultFullColorBitmap, new Int32Rect(roi.X, roi.Y, roi.Width, roi.Height));
            }
        }


        // ROI 표시 & 템플릿 매칭 ──────────────────────────────────────────────────
        private void ImageGridMouseDown(object sender, MouseButtonEventArgs e) // 좌우클릭 위치를 이미지 좌표로 변환해 저장
        {
            if (mainViewModel.IsBusy || (e.ChangedButton != MouseButton.Left && e.ChangedButton != MouseButton.Right)) { return; }

            ColorImage? image = mainViewModel.SourceImage;
            if (image == null) { return; }

            Grid grid = (Grid)sender;
            (double scale, double imageLeft, double imageTop) = GetUniformImageMetrics(grid, image);
            double x = e.GetPosition(grid).X - imageLeft;
            double y = e.GetPosition(grid).Y - imageTop;

            if (x < 0 || x > image.Width * scale || y < 0 || y > image.Height * scale) { return; }

            int imageX = (int)Math.Round(x / scale);
            int imageY = (int)Math.Round(y / scale);

            if (e.ChangedButton == MouseButton.Left)
            {
                leftClickX = imageX;
                leftClickY = imageY;
            }
            else
            {
                rightClickX = imageX;
                rightClickY = imageY;
            }

            UpdateRoiFromClicks();
            e.Handled = true;
        }
        private void ImageGridSizeChanged(object sender, SizeChangedEventArgs e) // 뷰어 크기가 바뀌면 ROI와 매칭 박스 다시 배치
        {
            UpdateRoiOverlay(SourceImageGrid, SourceRoiRectangle);
            UpdateRoiOverlay(ResultImageGrid, ResultRoiRectangle);
            UpdateTemplateMatchOverlay();
        }
        private void UpdateRoiFromClicks() // 두 클릭 좌표로 ROI를 만들고, 빈 영역이면 ROI를 해제
        {
            if (leftClickX < 0 || rightClickX < 0) { return; }

            Roi roi = Roi.FromPoints(leftClickX, leftClickY, rightClickX, rightClickY);

            if (roi.IsEmpty) { mainViewModel.ClearRoi(); }
            else { mainViewModel.SetRoi(roi); }
        }
        private void UpdateRoiOverlay(Grid grid, Rectangle rectangle) // ROI 박스를 화면에 표시
        {
            Roi roi = mainViewModel.CurrentRoi;
            ColorImage? image = mainViewModel.SourceImage;

            if (roi.IsEmpty || image == null || grid.ActualWidth <= 0 || grid.ActualHeight <= 0)
            {
                rectangle.Visibility = Visibility.Collapsed;
                return;
            }

            (double scale, double imageLeft, double imageTop) = GetUniformImageMetrics(grid, image);

            rectangle.Margin = new Thickness(imageLeft + roi.X * scale, imageTop + roi.Y * scale, 0, 0);
            rectangle.Width = roi.Width * scale;
            rectangle.Height = roi.Height * scale;
            rectangle.Visibility = Visibility.Visible;
        }
        private void ClearClicks() // ROI클릭 좌표를 초기화(새 이미지를 열거나 초기화할 때)
        {
            leftClickX = -1;
            leftClickY = -1;
            rightClickX = -1;
            rightClickY = -1;
        }
        private static (double scale, double imageLeft, double imageTop) GetUniformImageMetrics(Grid grid, ColorImage image) // Stretch=Uniform으로 표시된 이미지의 배율과 letterbox 여백을 계산
        {
            double scale = Math.Min(grid.ActualWidth / image.Width, grid.ActualHeight / image.Height);
            double imageLeft = (grid.ActualWidth - image.Width * scale) / 2;
            double imageTop = (grid.ActualHeight - image.Height * scale) / 2;

            return (scale, imageLeft, imageTop);
        }
        private void UpdateTemplateMatchOverlay() // ResultImageGrid 위에 TemplateMatches 후보 박스를 다시 그림
        {
            TemplateMatchCanvas.Children.Clear();

            ColorImage? image = mainViewModel.ResultImage;
            long[][] matches = mainViewModel.TemplateMatches;

            if (image == null || matches.Length == 0 || ResultImageGrid.ActualWidth <= 0 || ResultImageGrid.ActualHeight <= 0) { return; }

            (double scale, double imageLeft, double imageTop) = GetUniformImageMetrics(ResultImageGrid, image);
            Style boxStyle = (Style)FindResource("TemplateMatchRectangleStyle");

            foreach (long[] match in matches) // match = { x, y, 폭, 높이, 점수, 최대 점수 }
            {
                long x = match[0];
                long y = match[1];
                long width = match[2];
                long height = match[3];

                Rectangle box = new Rectangle { Style = boxStyle, Width = width * scale, Height = height * scale };

                Canvas.SetLeft(box, imageLeft + x * scale);
                Canvas.SetTop(box, imageTop + y * scale);

                TemplateMatchCanvas.Children.Add(box);
            }
        }
    }
}
