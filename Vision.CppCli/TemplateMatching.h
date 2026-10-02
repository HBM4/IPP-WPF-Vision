#pragma once

namespace Vision::CppCli
{
    public ref class TemplateMatching
    {
    public:
        static array<array<System::Int64>^>^ MatchDiff(Vision::Core::ColorImage^ source, Vision::Core::ColorImage^ result, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected);
        static array<array<System::Int64>^>^ MatchCorr(Vision::Core::ColorImage^ source, Vision::Core::ColorImage^ result, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected);
        static array<array<System::Int64>^>^ MatchCoeff(Vision::Core::ColorImage^ source, Vision::Core::ColorImage^ result, Vision::Core::Roi^ roi, bool isRedSelected, bool isGreenSelected, bool isBlueSelected);
    };
}
