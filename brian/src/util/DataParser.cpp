#include "brian/util/DataParser.h"

#include <regex>

namespace bb {

    Optional<DescriptorToken> parseToken(std::string token) {
        if (token.length() < 2) {
            return Optional<DescriptorToken>();
        }
        char c = token.at(0);
        if (c != 'o' && c != 'i') {
            return Optional<DescriptorToken>();
        }

        return Optional<DescriptorToken>(c);
    }

    std::vector<DeterministicSample> ParseDeterministicSamples(const std::string& filePath) {
        std::ifstream stream(filePath);
        if (stream.is_open()) {
            return ParseDeterministicSamples(stream);
        } else {
            return std::vector<DeterministicSample>();
        }
    }

    std::vector<DeterministicSample> ParseDeterministicSamples(std::ifstream& stream) {
        std::string line;
        std::string delim = ",";

        std::getline(stream, line);
        auto ind = line.find(delim);
        uint numInput = std::stoi(line.substr(0, ind));
        line.erase(0, ind + 1);
        
    }
}