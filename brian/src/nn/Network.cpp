#include "brian/NN.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

namespace bb {

    Network::Network(const NetworkDescriptor p_desc): m_desc(p_desc) {
        std::shared_ptr<Layer> prev(nullptr);
        for (uint i = 0; i < this->m_desc.numLayers; ++i) {
            auto current = std::make_shared<Layer>(
                i,
                p_desc.layerSizes[i],
                i < p_desc.numLayers - 1 ? p_desc.layerSizes[i + 1] : -1,
                &LeakyReLU
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

    void Network::Load(const Real* const p_input) {
        this->m_head->Load(p_input);
    }

    const Real* const Network::Feedforward() {
        this->m_tail->Feedforward();
        return this->m_tail->getNodeValues();
    }

}
