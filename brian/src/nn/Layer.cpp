#include "brian/NN.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

namespace bb {

    Layer::Layer(uint p_layerNum, ushort p_fromSize, ushort p_toSize, RActivation p_activationFunc):
    m_layerNum(p_layerNum), m_fromSize(p_fromSize), m_toSize(p_toSize), m_activationFunc(p_activationFunc) {
        this->m_next = nullptr;
        this->m_prev = nullptr;

        this->m_nodes = std::make_shared<Matrix>(this->m_fromSize, 1);
        
        // Exclude last layer from weights and biases since they are used for the next layer.
        if (p_toSize > 0) {
            this->m_weights = std::make_shared<Matrix>(p_toSize, p_fromSize);
            this->m_biases = std::make_shared<Matrix>(this->m_toSize, 1);
        }
    }

    Layer::~Layer() { }

    void Layer::Feedforward() {
        // Ensure last layer loads your nodes.
        if (this->m_prev) {
            this->m_prev->Feedforward();
        }

        // If this is the last layer, nothing more is needed for now.
        if (!this->m_next) {
            return;
        }

        this->m_weights->Mult(*this->m_nodes, *this->m_next->m_nodes);
        this->m_next->m_nodes->Add(*this->m_biases, *this->m_next->m_nodes);
        this->m_next->m_nodes->Mutate(this->m_next->m_activationFunc);
    }

    void Layer::Load(const Real* const p_values) {
        this->m_nodes->setAll(p_values);
    }

    void Layer::RandomizeWeights(const Real p_min, const Real p_max) {
        assert(p_max >= p_min);
        if (!m_weights) {
            return;
        }

        srand(time(0));
        const Real range = p_max - p_min;
        for (uint r = 0; r < this->m_weights->getNumRows(); ++r) {
            for (uint c = 0; c < this->m_weights->getNumCols(); ++c) {
                this->m_weights->set(r, c, rrand(p_min, p_max));
            }
        }
    }

    void Layer::RandomizeBiases(const Real p_min, const Real p_max) {
        assert(p_max >= p_min);
        if (!m_biases) {
            return;
        }

        srand(time(0));
        const Real range = p_max - p_min;
        for (uint r = 0; r < this->m_biases->getNumRows(); ++r) {
            this->m_biases->set(r, 0, rrand(p_min, p_max));
        }
    }

}