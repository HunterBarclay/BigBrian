#include "brian/nn.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <sstream>

namespace bb {

    Network::Network(const NetworkDescriptor desc): m_desc(desc) {
        std::shared_ptr<Layer> prev(nullptr);
        for (uint i = 0; i < this->m_desc.numLayers; ++i) {
            // Not sure the ternaries are the best choice here.
            std::shared_ptr<Layer> current;
            if (i < desc.numLayers - 1) {
                current = std::make_shared<Layer>(
                    i,
                    desc.layerSizes[i],
                    desc.layerSizes[i + 1],
                    desc.hiddenActivation,
                    desc.dHiddenActivation
                );
            } else {
                current = std::make_shared<Layer>(
                    i,
                    desc.layerSizes[i],
                    0,
                    desc.outputActivation,
                    desc.dOutputActivation
                );
            }
            current->set_prev(prev);
            if (prev) {
                prev->set_next(current);
            } else {
                this->m_head = current;
            }
            prev = current;
        }
        this->m_tail = prev;
    }

    Network::~Network() {
        std::cout << "Bye.\n";
    }

    void Network::randomize(const Real minBias, const Real maxBias, const Real minWeight, const Real maxWeight) {
        auto head = this->m_head;

        while (head) {
            head->randomize_biases(minBias, maxBias);
            head->randomize_weights(minWeight, maxWeight);
            head = head->get_next();
        }
    }

    void Network::load(const Real* const input) {
        this->m_head->load(input);
    }

    std::vector<Real> Network::feedforward() {
        this->m_tail->feedforward();
        auto vec = std::vector<Real>((size_t)this->m_tail->get_num_nodes());
        vec.assign(this->m_tail->get_num_nodes(), *this->m_tail->get_node_values());
        return std::move(vec);
    }

    void Network::back_propagate(const NetworkScore& scores) {
        this->m_tail->back_propagate(scores);
    }

    NetworkScore Network::score(const Real* const expected) const {
        std::vector<Real> nodeScores(this->m_tail->get_num_nodes());
        std::vector<Real> expectedVec(this->m_tail->get_num_nodes());
        auto nodeValues = this->m_tail->get_node_values();
        Real totalScore = 0;
        for (uint i = 0; i < this->m_tail->get_num_nodes(); ++i) {
            auto score = (nodeValues[i] - expected[i]) * (nodeValues[i] - expected[i]);
            expectedVec.at(i) = expected[i];
            nodeScores.at(i) = score;
            totalScore += score;
        }
        return {
            std::move(expectedVec),
            std::move(nodeScores),
            std::move(totalScore)
        };
    }

    void Network::train(const uint samples, const Real coef) {
        this->m_tail->train(samples, coef);
    }

    void Network::reset_training() {
        this->m_tail->reset_training();
    }

    std::string Network::str(bool input, bool hidden, bool weights, bool biases, bool output, bool derivs, bool derivAccums) const {
        std::stringstream ss;

        if (input) {
            ss << "\t[ INPUT ]\n";
            ss << this->m_head->get_nodes()->str();
        }

        if (hidden) {
            auto head = this->m_head->get_next();
            while (head) {
                if (biases) {
                    ss << "\t[ BIASES (" << head->get_layer_num() - 1 << ") ]\n";
                    ss << head->get_prev()->get_biases()->str();

                    if (derivs) {
                        ss << "\t[ Deriv BIASES (" << head->get_layer_num() - 1 << ") ]\n";
                        ss << head->get_prev()->get_d_biases()->str();
                    }

                    if (derivAccums) {
                        ss << "\t[ Deriv Accum BIASES (" << head->get_layer_num() - 1 << ") ]\n";
                        ss << head->get_prev()->get_d_biases_accum()->str();
                    }
                }
                if (weights) {
                    ss << "\t[ WEIGHTS (" << head->get_layer_num() - 1 << ") ]\n";
                    ss << head->get_prev()->get_weights()->str();

                    if (derivs) {
                        ss << "\t[ Deriv WEIGHTS (" << head->get_layer_num() - 1 << ") ]\n";
                        ss << head->get_prev()->get_d_weights()->str();
                    }

                    if (derivAccums) {
                        ss << "\t[ Deriv Accum WEIGHTS (" << head->get_layer_num() - 1 << ") ]\n";
                        ss << head->get_prev()->get_d_weights_accum()->str();
                    }
                }

                if (head->get_next()) {
                    ss << "\t[ HIDDEN (" << head->get_layer_num() << ") ]\n";
                    ss << head->get_nodes()->str();
                }

                head = head->get_next();
            }
        }

        if (output) {
            ss << "\t[ OUTPUT ]\n";
            ss << this->m_tail->get_nodes()->str();
        }

        return ss.str();
    }

}
