#include "brian/population.h"

#include <cstring>
#include <cassert>

namespace bb {

    DeterministicSample::DeterministicSample(const size_t numInput, const size_t numOutput, const Real* const input, const Real* const output) {
        this->m_numInput = numInput;
        this->m_numOutput = numOutput;
        this->m_input = new Real[numInput];
        this->m_output = new Real[numOutput];
        memcpy(this->m_input, input, sizeof(Real) * numInput);
        memcpy(this->m_output, output, sizeof(Real) * numOutput);
    }

    DeterministicSample::DeterministicSample(const DeterministicSample& sample) {
        this->m_numInput = sample.m_numInput;
        this->m_numOutput = sample.m_numOutput;
        this->m_input = new Real[sample.m_numInput];
        this->m_output = new Real[sample.m_numOutput];
        std::memcpy(this->m_input, sample.m_input, sizeof(Real) * sample.m_numInput);
        std::memcpy(this->m_output, sample.m_output, sizeof(Real) * sample.m_numOutput);
    }

    DeterministicSample::~DeterministicSample() {
        if (this->m_input != nullptr) delete[] this->m_input;
        if (this->m_output != nullptr) delete[] this->m_output;
    }

    DeterministicPopulation::DeterministicPopulation(const uint populationSize, const NetworkDescriptor desc): m_population(populationSize), m_iterations(0) {
        for (uint i = 0; i < populationSize; ++i) {
            auto n = std::make_shared<Network>(desc);
            n->randomize(-1, 1, -2, 2);
            m_population.at(i) = std::move(n);
        }
    }

    DeterministicPopulation::~DeterministicPopulation() { }

    void DeterministicPopulation::push_sample(const DeterministicSample& sample) {
        this->m_samples.push_back(DeterministicSample(sample));
    }

    void DeterministicPopulation::iterate(bool verbose) {
        for (auto networkIter = this->m_population.begin(); networkIter != this->m_population.end(); ++networkIter) {
            std::shared_ptr<Network> network = *networkIter;
            for (auto sampleIter = this->m_samples.begin(); sampleIter != this->m_samples.end(); ++sampleIter) {
                network->load((*sampleIter).input());
                network->feedforward();
                NetworkScore scores = network->score((*sampleIter).output());
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
                network->load((*sampleIter).input());
                network->feedforward();
                NetworkScore scores = network->score((*sampleIter).output());
                scoreAccum += scores.overallScore;
            }
        }
        return scoreAccum / this->m_population.size();
    }

}