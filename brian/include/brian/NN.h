#pragma once
/**
 * @file NN.h
 * @author Hunter Barclay
 * @brief Really simple, conentional neural network.
 * 
 * https://www.v7labs.com/blog/neural-networks-activation-functions
 * 
 * First activation functions I'll try using:
 * Hidden Layers: Leaky ReLU
 * Output Layer: Linear
 */

#include "math/Matrix.h"

#include <iostream>
#include <memory>
#include <vector>

namespace bb {

    Real LeakyReLU(Real a);
    Real dLeakyReLU(Real a);

    Real Linear(Real a);
    Real dLinear(Real a);

    struct NetworkDescriptor {
        const uint numLayers;
        const ushort *layerSizes;
    };

    /**
     * @brief A layer of nodes, with references to previous and next layers, as well
     * as weights to propagate to the next.
     *
     * This effectively acts as a node in a doubly linked list.
     */
    class Layer {
    private:
        std::unique_ptr<bb::Matrix> m_biases;
        std::unique_ptr<bb::Matrix> m_weights;

        std::shared_ptr<bb::Matrix> m_nodes;

        std::shared_ptr<Layer> m_next;
        std::shared_ptr<Layer> m_prev;

        const uint m_num;
        const ushort m_fromSize;
        const ushort m_toSize;

        const RActivation m_activationFunc;
    public:
        /**
         * @brief Construct a new Layer object
         *
         * @param p_layerNum
         * @param p_fromSize
         * @param p_toSize
         */
        Layer(uint p_layerNum, ushort p_fromSize, ushort p_toSize, RActivation p_activationFunc);
        Layer(const Layer &_) = delete;
        ~Layer();

        /**
         * @brief Set the value of the nodes in the layer.
         * 
         * @param p_values Values to assign to the nodes of the layer. Must be the same
         * size as the layer.
         */
        void Load(const Real *const p_values);

        void Feedforward();

        /**
         * @brief Randomize the weights in the layer.
         * 
         * Nothing happens if this is the last layer. Will fail if
         * max is less than min.
         * 
         * @param p_min Minimum a weight can be.
         * @param p_max Maximum a weight can be.
         */
        void randomizeWeights(const Real p_min, const Real p_max);

        inline const Real* const getNodeValues() const {
            return this->m_nodes->getRef();
        }

        /**
         * @brief Set the next layer.
         *
         * @param p_next Next layer.
         */
        inline void setNext(std::shared_ptr<Layer> p_next) {
            this->m_next = p_next;
        }

        /**
         * @brief Set the Prev object
         *
         * @param p_prev
         */
        inline void setPrev(std::shared_ptr<Layer> p_prev) {
            this->m_prev = p_prev;
        }
    };

    /**
     * @brief Neural network, consisting of layers of nodes that propagate from an input
     * layer to an output layer.
     */
    class Network {
    private:
        std::shared_ptr<Layer> m_head;
        std::shared_ptr<Layer> m_tail;

        const NetworkDescriptor m_desc;
    public:
        /**
         * @brief Construct a new Network object
         *
         * @param p_desc Descriptor of the network.
         */
        Network(const NetworkDescriptor m_desc);
        Network(const Network &_) = delete;
        ~Network();

        void Load(const Real* const p_input);

        const Real* const Feedforward();
    };
}
