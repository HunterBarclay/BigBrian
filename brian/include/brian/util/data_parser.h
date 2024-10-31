#pragma once

#include "brian/population.h"

#include <iostream>
#include <fstream>
#include <string>

#include "brian/util/util.h"

namespace bb {
    typedef struct {
        size_t numIn;
        size_t numOut;
    } DeterministicCSVDescriptor;

    Optional<std::vector<DeterministicSample>> parse_deterministic_samples(const std::string& filePath);
    Optional<std::vector<DeterministicSample>> parse_deterministic_samples(std::istream& stream);
}
