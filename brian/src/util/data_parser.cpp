#include "brian/util/data_parser.h"

#include <regex>

namespace bb {

    std::vector<std::string> tokenize_line(const std::string& line, const std::string& delim) {
        size_t i = 0;
        size_t j;
        std::vector<std::string> tokens;
        while ((j = line.find(delim, i)) != std::string::npos) {
            std::string token = line.substr(i, j - i);
            if (token.size() > 0) tokens.push_back(token);
            i = j + delim.size();
        }
        if (i < line.length() - 1) tokens.push_back(line.substr(i));
        return std::move(tokens);
    }

    Optional<DeterministicCSVDescriptor> validate_labels(const std::vector<std::string>& labels) {
        size_t numIn = 0;
        size_t numOut = 0;
        char inLabel[1 + 20], outLabel[1 + 20];
        bool readingIn = false;
        sprintf(inLabel, "i%lu", numIn);
        sprintf(inLabel, "o%lu", numOut);
        for (auto iter = labels.begin(); iter != labels.end(); ++iter) {
            if (iter->length() == 0) {
                return unavailable<DeterministicCSVDescriptor>();
            }

            if (readingIn) {
                if (!strcmp(iter->c_str(), inLabel)) {
                    ++numIn;
                    sprintf(inLabel, "i%lu", numIn);
                } else if (!strcmp(iter->c_str(), outLabel)) {
                    ++numOut;
                    sprintf(inLabel, "o%lu", numOut);
                    readingIn = false;
                } else {
                    return unavailable<DeterministicCSVDescriptor>();
                }
            } else {
                if (!strcmp(iter->c_str(), outLabel)) {
                    ++numOut;
                    sprintf(inLabel, "o%lu", numOut);
                } else {
                    return unavailable<DeterministicCSVDescriptor>();
                }
            }
        }

        return available<DeterministicCSVDescriptor>({ numIn, numOut });
    }

    Optional<std::vector<DeterministicSample>> parse_deterministic_samples(const std::string& filePath) {
        std::ifstream stream(filePath);
        if (stream.is_open()) {
            return parse_deterministic_samples(stream);
        } else {
            return unavailable<std::vector<DeterministicSample>>();
        }
    }

    Optional<std::vector<DeterministicSample>> parse_deterministic_samples(std::ifstream& stream) {
        std::string line;
        std::string delim = ",";

        std::getline(stream, line);
        auto labels = tokenize_line(line, delim);
        auto descriptor = validate_labels(labels);
        if (descriptor.has()) {
            return unavailable<std::vector<DeterministicSample>>();
        }


    }
}