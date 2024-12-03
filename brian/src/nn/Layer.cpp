#include "brian/nn.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

namespace bb {

    Layer::Layer(uint layerNum, ushort fromSize, ushort toSize, RActivation activationFunc, RActivation dActivationFunc):
    m_layerNum(layerNum), m_fromSize(fromSize), m_toSize(toSize), m_activationFunc(activationFunc), m_dActivationFunc(dActivationFunc) {
        this->m_next = nullptr;
        this->m_prev = nullptr;

        this->m_nodes = std::make_shared<Matrix>(this->m_fromSize, 1);
        this->m_preactivation = std::make_shared<Matrix>(this->m_fromSize, 1);
        this->m_dPreactivation = std::make_shared<Matrix>(this->m_fromSize, 1);
        
        // Exclude last layer from weights and biases since they are used for the next layer.
        if (toSize > 0) {
            this->m_weights = std::make_shared<Matrix>(toSize, fromSize);
            this->m_dWeights = std::make_shared<Matrix>(toSize, fromSize);
            this->m_dWeightsAccum = std::make_shared<Matrix>(toSize, fromSize);
            this->m_biases = std::make_shared<Matrix>(toSize, 1);
            this->m_dBiases = std::make_shared<Matrix>(toSize, 1);
            this->m_dBiasesAccum = std::make_shared<Matrix>(toSize, 1);
        }
    }

    Layer::~Layer() { }

    void Layer::feedforward() {
        // Ensure last layer loads your nodes.
        if (this->m_prev) {
            this->m_prev->feedforward();
        }

        // If this is the last layer, nothing more is needed for now.
        if (!this->m_next) {
            return;
        }

        this->m_weights->mult(*this->m_nodes, *this->m_next->m_preactivation);
        this->m_next->m_preactivation->add(*this->m_biases, *this->m_next->m_preactivation);
        this->m_next->m_preactivation->copy_to(*this->m_next->m_nodes);
        this->m_next->m_nodes->mutate(this->m_next->m_activationFunc);
    }

    void Layer::back_propagate() {
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

        this->m_dBiasesAccum->add(*this->m_dBiases, *this->m_dBiasesAccum);
        this->m_dWeightsAccum->add(*this->m_dWeights, *this->m_dWeightsAccum);

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
            this->m_prev->back_propagate();
        }
    }

    void Layer::back_propagate(const NetworkScore& scores) {
        assert(!this->m_dWeights && !this->m_dBiases && this->m_prev);

        for (uint i = 0; i < scores.nodeScores.size(); ++i) {
            this->m_dPreactivation->set(i, 0,
                2 * (this->m_nodes->get(i, 0) - scores.expected.at(i)) // d[cost] / d[node]
                * this->m_dActivationFunc(this->m_preactivation->get(i, 0)) // d[node] / d[preactivation]
            );
        }

        this->m_prev->back_propagate();
    }

    void Layer::train(const uint samples, const Real coef) {
        if (this->m_dWeightsAccum && this->m_dBiasesAccum) {
            m_dWeightsAccum->mult(1.0 / samples);
            m_dBiasesAccum->mult(1.0 / samples);
            for (ushort toNode = 0; toNode < this->m_toSize; ++toNode) {
                Real bias = this->m_biases->get(toNode, 0);
                this->m_biases->set(toNode, 0, bias - this->m_dBiasesAccum->get(toNode, 0) * coef);
                for (ushort fromNode = 0; fromNode < this->m_fromSize; ++fromNode) {
                    Real weight = this->m_weights->get(toNode, fromNode);
                    this->m_weights->set(toNode, fromNode, weight - this->m_dWeightsAccum->get(toNode, fromNode) * coef);
                }
            }
        }

        if (this->m_prev) {
            this->m_prev->train(samples, coef);
        }
    }

    void Layer::reset_training() {
        if (this->m_toSize > 0) {
            this->m_dBiases->clear();
            this->m_dBiasesAccum->clear();
            this->m_dWeights->clear();
            this->m_dWeightsAccum->clear();
        }

        if (this->m_prev) {
            this->m_prev->reset_training();
        }
    }

    void Layer::load(const Real* const values) {
        this->m_nodes->set_all(values);
    }

    void Layer::randomize_weights(const Real min, const Real max) {
        assert(max >= min);
        if (!m_weights) {
            return;
        }

        srand(time(0));
        const Real range = max - min;
        for (uint r = 0; r < this->m_weights->get_num_rows(); ++r) {
            for (uint c = 0; c < this->m_weights->get_num_cols(); ++c) {
                this->m_weights->set(r, c, rrand(min, max));
            }
        }
    }

    void Layer::randomize_biases(const Real min, const Real max) {
        assert(max >= min);
        if (!m_biases) {
            return;
        }

        srand(time(0));
        const Real range = max - min;
        for (uint r = 0; r < this->m_biases->get_num_rows(); ++r) {
            this->m_biases->set(r, 0, rrand(min, max));
        }
    }

}