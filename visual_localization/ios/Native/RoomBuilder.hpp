#pragma once
#include <functional>
#include <string>

namespace roomloc {
// All paths are local. Outputs are compatible with desktop .roomloc maps.
// Returning false cancels. A negative fraction is a cancellation-only poll.
using BuildProgress = std::function<bool(double, const std::string&)>;
struct BuildOptions {
    std::string name = "Room";
    int maxReferences = 120;
    int featuresPerImage = 1500;
};
void buildRoom(const std::string& glbPath, const std::string& referenceRoot,
               const std::string& outputDirectory, const BuildOptions& options = {},
               const BuildProgress& progress = {});
}
