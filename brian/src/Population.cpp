#include "brian/Population.h"

#include <cassert>

namespace bb {

    Population::Population(const uint p_populationSize, const NetworkDescriptor p_desc): m_population(p_populationSize), m_iterations(0) {
        for (uint i = 0; i < p_populationSize; ++i) {
            auto n = std::make_shared<Network>(p_desc);
            n->Randomize(-10, 10, -5, 5);
            m_population.at(i) = std::move(n);
        }
    }

    Population::~Population() { }

    void Population::PushSample(const DeterministicSample& p_sample) {
        this->m_samples.push_back(p_sample);
    }

    void Population::Iterate(bool p_verbose) {
        for (auto networkIter = this->m_population.begin(); networkIter != this->m_population.end(); ++networkIter) {
            std::shared_ptr<Network> network = *networkIter;
            for (auto sampleIter = this->m_samples.begin(); sampleIter != this->m_samples.end(); ++sampleIter) {
                network->Load((*sampleIter).input);
                network->Feedforward();
                NetworkScore scores = network->Score((*sampleIter).output);
                network->BackPropagate(scores);

                if (p_verbose) {
                    std::cout << "\nNetwork Results:\n";
                    std::cout << network->str(true, false, false, false, true, false, false);
                }
            }
            network->Train(0.01);
            network->ResetTraining();
        }
    }

    Real Population::getAverageScore() {
        Real scoreAccum = 0;
        for (auto networkIter = this->m_population.begin(); networkIter != this->m_population.end(); ++networkIter) {
            std::shared_ptr<Network> network = *networkIter;
            for (auto sampleIter = this->m_samples.begin(); sampleIter != this->m_samples.end(); ++sampleIter) {
                network->Load((*sampleIter).input);
                network->Feedforward();
                NetworkScore scores = network->Score((*sampleIter).output);
                scoreAccum += scores.overallScore;
            }
        }
        return scoreAccum / this->m_population.size();
    }

}