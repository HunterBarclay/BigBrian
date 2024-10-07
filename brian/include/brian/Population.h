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

    class Population {
    private:
        std::vector<std::shared_ptr<Network>> m_population;
        std::vector<DeterministicSample> m_samples;
    public:
        Population(const uint p_populationSize, const NetworkDescriptor p_desc);
        Population(const Population& _) = delete;
        ~Population();

        void PushSample(const DeterministicSample& p_sample);
    };
}