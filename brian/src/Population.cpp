#include "brian/Population.h"

namespace bb {

    Population::Population(const uint p_populationSize, const NetworkDescriptor p_desc): m_population(p_populationSize) {
        for (int i = 0; i < p_populationSize; ++i) {
            m_population.push_back(std::make_shared<Network>(p_desc));
        }
    }

    Population::~Population() { }

    void Population::PushSample(const DeterministicSample& p_sample) {
        this->m_samples.push_back(p_sample);
    }

}