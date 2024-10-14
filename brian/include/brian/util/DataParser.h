#pragma once

#include "brian/Population.h"

#include <iostream>
#include <fstream>
#include <string>

#include "brian/util/util.h"

namespace bb {
    using DescriptorToken = char;

    std::vector<DeterministicSample> ParseDeterministicSamples(const std::string& filePath);
    std::vector<DeterministicSample> ParseDeterministicSamples(std::ifstream& stream);
}
