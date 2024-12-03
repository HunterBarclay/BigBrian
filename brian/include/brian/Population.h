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
    class DeterministicSample {
    private:
        size_t m_numInput;
        size_t m_numOutput;
        Real* m_input;
        Real* m_output;
    public:
        DeterministicSample(const size_t numInput, const size_t numOutput, const Real* input, const Real* output);
        DeterministicSample(const DeterministicSample& sample);
        ~DeterministicSample();

        inline const size_t numInput() { return this->m_numInput; }
        inline const size_t numOutput() { return this->m_numOutput; }
        inline const Real* const input() { return this->m_input; }
        inline const Real* const output() { return this->m_output; }
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