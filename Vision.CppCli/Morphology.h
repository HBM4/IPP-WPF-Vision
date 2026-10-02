#pragma once

namespace Vision::CppCli
{
    public ref class Morphology
    {
    public:
        static void Dilate(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize);
        static void Erode(Vision::Core::ColorImage^ image, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected, int kernelSize);
    };
}
