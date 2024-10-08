#include "brian/NN.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <sstream>

namespace bb {

    Network::Network(const NetworkDescriptor p_desc): m_desc(p_desc) {
        std::shared_ptr<Layer> prev(nullptr);
        for (uint i = 0; i < this->m_desc.numLayers; ++i) {
            // Not sure the ternaries are the best choice here.
            auto current = std::make_shared<Layer>(
                i,
                p_desc.layerSizes[i],
                i < p_desc.numLayers - 1 ? p_desc.layerSizes[i + 1] : -1,
                i < p_desc.numLayers - 1 ? p_desc.hiddenActivation : p_desc.outputActivation
            );
            current->setPrev(prev);
            if (prev) {
                prev->setNext(current);
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

    void Network::Randomize(const Real p_minBias, const Real p_maxBias, const Real p_minWeight, const Real p_maxWeight) {
        auto head = this->m_head;

        while (head) {
            head->RandomizeBiases(p_minBias, p_maxBias);
            head->RandomizeWeights(p_minWeight, p_maxWeight);
            head = head->getNext();
        }
    }

    void Network::Load(const Real* const p_input) {
        this->m_head->Load(p_input);
    }

    std::vector<Real> Network::Feedforward() {
        this->m_tail->Feedforward();
        auto vec = std::vector<Real>((size_t)this->m_tail->getNumNodes());
        vec.assign(this->m_tail->getNumNodes(), *this->m_tail->getNodeValues());
        return std::move(vec);
    }

    NetworkScore Network::Score(const Real* const p_expected) const {
        std::vector<Real> nodeScores(this->m_tail->getNumNodes());
        auto nodeValues = this->m_tail->getNodeValues();
        Real totalScore = 0;
        for (uint i = 0; i < this->m_tail->getNumNodes(); ++i) {
            auto score = (nodeValues[i] - p_expected[i]) * (nodeValues[i] - p_expected[i]);
            nodeScores.push_back(score);
            totalScore += score;
        }
        return {
            std::move(nodeScores),
            std::move(totalScore)
        };
    }

    std::string Network::str(bool p_input, bool p_hidden, bool p_weights, bool p_biases, bool p_output) const {
        std::stringstream ss;

        if (p_input) {
            ss << "\t[ INPUT ]\n";
            ss << this->m_head->getNodes()->str();
        }

        if (p_hidden) {
            auto head = this->m_head->getNext();
            while (head) {
                if (p_biases) {
                    ss << "\t[ BIASES (" << head->getLayerNum() - 1 << ") ]\n";
                    ss << head->getPrev()->getBiases()->str();
                }
                if (p_weights) {
                    ss << "\t[ WEIGHTS (" << head->getLayerNum() - 1 << ") ]\n";
                    ss << head->getPrev()->getWeights()->str();
                }

                if (head->getNext()) {
                    ss << "\t[ HIDDEN (" << head->getLayerNum() << ") ]\n";
                    ss << head->getNodes()->str();
                }

                head = head->getNext();
            }
        }

        if (p_output) {
            ss << "\t[ OUTPUT ]\n";
            ss << this->m_tail->getNodes()->str();
        }

        return ss.str();
    }

}
