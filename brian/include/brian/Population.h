#pragma once
/**
 * @file Population.h
 * @author Hunter Barclay
 * @brief Population for handling a number of networks at once.
 * 
 * For right now, each population will be started with random traits (weights/biases).
 */

#include <vector>

#include "brian/NN.h"

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
        DeterministicPopulation(const uint p_populationSize, const NetworkDescriptor p_desc);
        DeterministicPopulation(const DeterministicPopulation& _) = delete;
        ~DeterministicPopulation();

        void PushSample(const DeterministicSample& p_sample);
        void Iterate(bool p_verbose);
        Real getAverageScore();

        const inline uint getNumIterations() const {
            return this->m_iterations;
        }
    };
}