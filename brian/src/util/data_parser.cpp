#include "brian/util/data_parser.h"

#include <regex>

namespace bb {

    std::vector<std::string> tokenize_line(const std::string& line, const std::string& delim) {
        size_t i = 0;
        size_t j;
        std::vector<std::string> tokens;
        while ((j = line.find(delim, i)) != std::string::npos) {
            std::string token = line.substr(i, j - i);
            if (token.size() > 0) {
                printf("Token found: '%s'\n", token.c_str());
                tokens.push_back(token);
            }
            i = j + delim.size();
        }
        if (i <= line.length() - 1) {
            auto token = line.substr(i);
            printf("Final token: '%s'\n", token.c_str());
            tokens.push_back(token);
        } else {
            printf("Skipping final token: Len %d, i %d\n", line.length(), i);
        }
        return std::move(tokens);
    }

    Optional<DeterministicCSVDescriptor> validate_labels(const std::vector<std::string>& labels) {
        size_t numIn = 0;
        size_t numOut = 0;
        char inLabel[1 + 20], outLabel[1 + 20];
        bool readingIn = true;
        sprintf(inLabel, "i%lu", numIn);
        sprintf(outLabel, "o%lu", numOut);

        printf("Matchable in label: '%s'\n", inLabel);
        printf("Matchable out label: '%s'\n", outLabel);

        for (auto iter = labels.begin(); iter != labels.end(); ++iter) {
            if (iter->length() == 0) {
                printf("Empty token\n");
                return unavailable<DeterministicCSVDescriptor>();
            }

            printf("Parsing label token: '%s'\n", iter->c_str());

            if (readingIn) {
                if (!strcmp(iter->c_str(), inLabel)) {
                    printf("In label found\n");
                    ++numIn;
                    sprintf(inLabel, "i%lu", numIn);
                    printf("Matchable in label: '%s'\n", inLabel);
                } else if (!strcmp(iter->c_str(), outLabel)) {
                    printf("First Out label found\n");
                    ++numOut;
                    sprintf(outLabel, "o%lu", numOut);
                    printf("Matchable out label: '%s'\n", outLabel);
                    readingIn = false;
                } else {
                    printf("Invalid label found\n");
                    return unavailable<DeterministicCSVDescriptor>();
                }
            } else {
                if (!strcmp(iter->c_str(), outLabel)) {
                    printf("Out label found\n");
                    ++numOut;
                    sprintf(outLabel, "o%lu", numOut);
                    printf("Matchable out label: '%s'\n", outLabel);
                } else {
                    printf("Invalid label found\n");
                    return unavailable<DeterministicCSVDescriptor>();
                }
            }
        }

        printf("%d inputs, %d outputs\n", numIn, numOut);
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

    Optional<std::vector<DeterministicSample>> parse_deterministic_samples(std::istream& stream) {
        std::string line;
        std::string delim = ",";
        std::vector<DeterministicSample> samples;

        std::getline(stream, line);
        printf("Parsing labels: '%s'\n", line.c_str());
        auto labels = tokenize_line(line, delim);
        auto descriptorOp = validate_labels(labels);
        if (!descriptorOp.has()) {
            printf("Invalid labels\n");
            return unavailable<std::vector<DeterministicSample>>();
        }

        auto numIn = descriptorOp.value().numIn;
        auto numOut = descriptorOp.value().numOut;
        size_t columns = numIn + numOut;
        while (!stream.eof()) {
            std::getline(stream, line);

            if (stream.fail()) {
                break;
            }

            printf("Parsing row: '%s'\n", line.c_str());

            auto values = tokenize_line(line, delim);
            if (values.size() == 0) {
                printf("Row is empty\n");
                continue;
            } else if (values.size() != columns) {
                printf("Row has invalid number of columns: %d", values.size());
                return unavailable<std::vector<DeterministicSample>>();
            }

            auto input = new Real[numIn];
            auto output = new Real[numOut];

            for (size_t i = 0; i < numIn; ++i) {
                input[i] = from_str(values[i]);
            }
            for (size_t i = 0; i < numOut; ++i) {
                output[i] = from_str(values[numIn + i]);
            }

            samples.push_back(DeterministicSample(numIn, numOut, input, output));
        }

        printf("Done Parsing\n");

        return available<std::vector<DeterministicSample>>(std::move(samples));
    }
}