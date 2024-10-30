#pragma once
/**
 * @file Population.h
 * @author Hunter Barclay
 * @brief Population for handling a number of networks at once.
 * 
 * For right now, each population will be started with random traits (weights/biases).
 */

#include <vector>

#include "brian/nn.h"

namespace bb {
    /**
     * @brief Sample of deterministic data including input and expected output.
     */
    struct DeterministicSample {
        const Real* const input;
        const Real* const output;
    };

    class DeterministicPopulation {
    private:
        std::vector<std::shared_ptr<Network>> m_population;
        std::vector<DeterministicSample> m_samples;

        uint m_iterations;
    public:
        DeterministicPopulation(const uint populationSize, const NetworkDescriptor desc);
        DeterministicPopulation(const DeterministicPopulation& _) = delete;
        ~DeterministicPopulation();

        void push_sample(const DeterministicSample& sample);
        void iterate(bool verbose);
        Real get_average_score();

        const inline uint get_num_iterations() const {
            return this->m_iterations;
        }
    };
}