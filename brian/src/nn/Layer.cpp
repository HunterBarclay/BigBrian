#include "brian/NN.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

namespace bb {

    Layer::Layer(uint p_layerNum, ushort p_fromSize, ushort p_toSize, RActivation p_activationFunc, RActivation p_dActivationFunc):
    m_layerNum(p_layerNum), m_fromSize(p_fromSize), m_toSize(p_toSize), m_activationFunc(p_activationFunc), m_dActivationFunc(p_dActivationFunc) {
        this->m_next = nullptr;
        this->m_prev = nullptr;

        this->m_nodes = std::make_shared<Matrix>(this->m_fromSize, 1);
        this->m_preactivation = std::make_shared<Matrix>(this->m_fromSize, 1);
        this->m_dPreactivation = std::make_shared<Matrix>(this->m_fromSize, 1);
        
        // Exclude last layer from weights and biases since they are used for the next layer.
        if (p_toSize > 0) {
            this->m_weights = std::make_shared<Matrix>(p_toSize, p_fromSize);
            this->m_dWeights = std::make_shared<Matrix>(p_toSize, p_fromSize);
            this->m_dWeightsAccum = std::make_shared<Matrix>(p_toSize, p_fromSize);
            this->m_biases = std::make_shared<Matrix>(p_toSize, 1);
            this->m_dBiases = std::make_shared<Matrix>(p_toSize, 1);
            this->m_dBiasesAccum = std::make_shared<Matrix>(p_toSize, 1);
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

        this->m_weights->Mult(*this->m_nodes, *this->m_next->m_preactivation);
        this->m_next->m_preactivation->Add(*this->m_biases, *this->m_next->m_preactivation);
        this->m_next->m_preactivation->CopyTo(*this->m_next->m_nodes);
        this->m_next->m_nodes->Mutate(this->m_next->m_activationFunc);
    }

    void Layer::BackPropagate() {
        assert(this->m_dWeights && this->m_dBiases && this->m_next);

        for (ushort toNode = 0; toNode < this->m_toSize; ++toNode) {
            const Real preactivationValue = this->m_next->m_dPreactivation->get(toNode, 0);
            // d[preactivation] / d[bias]
            this->m_dBiases->set(toNode, 0,
                preactivationValue
            );
            for (ushort fromNode = 0; fromNode < this->m_fromSize; ++fromNode) {
                // d[preactivation] / d[weight]
                this->m_dWeights->set(toNode, fromNode,
                    this->m_nodes->get(fromNode, 0) * preactivationValue
                );
            }
        }

        this->m_dBiasesAccum->Add(*this->m_dBiases, *this->m_dBiasesAccum);
        this->m_dWeightsAccum->Add(*this->m_dWeights, *this->m_dWeightsAccum);

        for (ushort fromNode = 0; fromNode < this->m_fromSize; ++fromNode) {
            Real dNode = 0;
            for (ushort toNode = 0; toNode < this->m_toSize; ++toNode) {
                dNode += this->m_weights->get(toNode, fromNode) * this->m_next->m_dPreactivation->get(toNode, 0);
            }
            this->m_dPreactivation->set(fromNode, 0,
                dNode * this->m_dActivationFunc(this->m_preactivation->get(fromNode, 0)) // d[preactivation of next layer] / d[preactivation of this current]
            );
        }

        if (this->m_prev) {
            this->m_prev->BackPropagate();
        }
    }

    void Layer::BackPropagate(const NetworkScore& p_scores) {
        assert(!this->m_dWeights && !this->m_dBiases && this->m_prev);

        for (uint i = 0; i < p_scores.nodeScores.size(); ++i) {
            this->m_dPreactivation->set(i, 0,
                2 * (this->m_nodes->get(i, 0) - p_scores.expected.at(i)) // d[cost] / d[node]
                * this->m_dActivationFunc(this->m_preactivation->get(i, 0)) // d[node] / d[preactivation]
            );
        }

        this->m_prev->BackPropagate();
    }

    void Layer::Train(const uint p_samples, const Real p_coef) {
        if (this->m_dWeightsAccum && this->m_dBiasesAccum) {
            m_dWeightsAccum->Mult(1.0 / p_samples);
            m_dBiasesAccum->Mult(1.0 / p_samples);
            for (ushort toNode = 0; toNode < this->m_toSize; ++toNode) {
                Real bias = this->m_biases->get(toNode, 0);
                this->m_biases->set(toNode, 0, bias - this->m_dBiasesAccum->get(toNode, 0) * p_coef);
                for (ushort fromNode = 0; fromNode < this->m_fromSize; ++fromNode) {
                    Real weight = this->m_weights->get(toNode, fromNode);
                    this->m_weights->set(toNode, fromNode, weight - this->m_dWeightsAccum->get(toNode, fromNode) * p_coef);
                }
            }
        }

        if (this->m_prev) {
            this->m_prev->Train(p_samples, p_coef);
        }
    }

    void Layer::ResetTraining() {
        if (this->m_toSize > 0) {
            this->m_dBiases->Clear();
            this->m_dBiasesAccum->Clear();
            this->m_dWeights->Clear();
            this->m_dWeightsAccum->Clear();
        }

        if (this->m_prev) {
            this->m_prev->ResetTraining();
        }
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