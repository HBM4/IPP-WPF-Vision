#pragma once

namespace Vision::CppCli
{
    public ref class SpatialFilter
    {
    public:
        static void Gaussian(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize, double sigma);
        static void Laplacian(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize, double sigma);
        static void Sobel(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize);
    };
}
