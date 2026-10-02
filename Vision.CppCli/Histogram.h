#pragma once

namespace Vision::CppCli
{
    public ref class Histogram
    {
    public:
        static Vision::Core::HistogramData^ Calculate(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi);
        static void Binarize(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected);
        static void Equalize(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected);
    };
}
