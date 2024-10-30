#include "brian/population.h"

#include <cassert>

namespace bb {

    DeterministicPopulation::DeterministicPopulation(const uint populationSize, const NetworkDescriptor desc): m_population(populationSize), m_iterations(0) {
        for (uint i = 0; i < populationSize; ++i) {
            auto n = std::make_shared<Network>(desc);
            n->randomize(-1, 1, -2, 2);
            m_population.at(i) = std::move(n);
        }
    }

    DeterministicPopulation::~DeterministicPopulation() { }

    void DeterministicPopulation::push_sample(const DeterministicSample& sample) {
        this->m_samples.push_back(sample);
    }

    void DeterministicPopulation::iterate(bool verbose) {
        for (auto networkIter = this->m_population.begin(); networkIter != this->m_population.end(); ++networkIter) {
            std::shared_ptr<Network> network = *networkIter;
            for (auto sampleIter = this->m_samples.begin(); sampleIter != this->m_samples.end(); ++sampleIter) {
                network->load((*sampleIter).input);
                network->feedforward();
                NetworkScore scores = network->score((*sampleIter).output);
                network->back_propagate(scores);

                if (verbose) {
                    std::cout << "\nNetwork Results:\n";
                    std::cout << network->str(true, false, false, false, true, false, false);
                }
            }
            if (verbose) {
                std::cout << "Network:\n" << network->str(true, true, true, true, true, false, false);
            }
            network->train(this->m_samples.size(), 0.01);
            network->reset_training();
        }
        this->m_iterations++;
    }

    Real DeterministicPopulation::get_average_score() {
        Real scoreAccum = 0;
        for (auto networkIter = this->m_population.begin(); networkIter != this->m_population.end(); ++networkIter) {
            std::shared_ptr<Network> network = *networkIter;
            for (auto sampleIter = this->m_samples.begin(); sampleIter != this->m_samples.end(); ++sampleIter) {
                network->load((*sampleIter).input);
                network->feedforward();
                NetworkScore scores = network->score((*sampleIter).output);
                scoreAccum += scores.overallScore;
            }
        }
        return scoreAccum / this->m_population.size();
    }

}