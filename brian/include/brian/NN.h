#pragma once
/**
 * @file NN.h
 * @author Hunter Barclay
 * @brief Really simple, conentional neural network.
 * 
 * I really don't know why I'm using ushort for layer sizes.
 * 
 * https://www.v7labs.com/blog/neural-networks-activation-functions
 * 
 * First activation functions I'll try using:
 * Hidden Layers: Leaky ReLU
 * Output Layer: Linear
 */

#include "brian/math/matrix.h"

#include <iostream>
#include <memory>
#include <vector>

namespace bb {

    namespace activation {
        Real leaky_re_lu(Real a);
        Real d_leaky_re_lu(Real a);

        Real linear(Real a);
        Real d_linear(Real a);
    }

    Real Sigmoid(Real a);
    Real dSigmoid(Real a);

    /**
     * @brief Descriptor for a convolutional neural network.
     */
    struct NetworkDescriptor {
        const uint numLayers;
        const ushort *layerSizes;
        const RActivation hiddenActivation;
        const RActivation dHiddenActivation;
        const RActivation outputActivation;
        const RActivation dOutputActivation;
    };

    /**
     * @brief Score of a network.
     */
    struct NetworkScore {
        std::vector<Real> expected;
        std::vector<Real> nodeScores;
        const Real overallScore;
    };

    /**
     * @brief A layer of nodes, with references to previous and next layers, as well
     * as weights to propagate to the next.
     *
     * This effectively acts as a node in a doubly linked list.
     */
    class Layer {
    private:
        std::shared_ptr<bb::Matrix> m_biases;
        std::shared_ptr<bb::Matrix> m_weights;

        std::shared_ptr<bb::Matrix> m_nodes;
        std::shared_ptr<bb::Matrix> m_preactivation;

        std::shared_ptr<bb::Matrix> m_dBiases;
        std::shared_ptr<bb::Matrix> m_dWeights;
        std::shared_ptr<bb::Matrix> m_dPreactivation;
        std::shared_ptr<bb::Matrix> m_dBiasesAccum;
        std::shared_ptr<bb::Matrix> m_dWeightsAccum;

        std::shared_ptr<Layer> m_next;
        std::shared_ptr<Layer> m_prev;

        const uint m_layerNum;
        const ushort m_fromSize;
        const ushort m_toSize;

        const RActivation m_activationFunc;
        const RActivation m_dActivationFunc;

        /**
         * @brief Back propagates through the layers, generating derivatives for
         * all inputs along the way.
         */
        void back_propagate();
    public:
        /**
         * @brief Construct a new Layer object
         *
         * @param layerNum
         * @param fromSize
         * @param toSize
         */
        Layer(uint layerNum, ushort fromSize, ushort toSize, RActivation activationFunc, RActivation dActivationFunc);
        Layer(const Layer &_) = delete;
        ~Layer();

        /**
         * @brief Set the value of the nodes in the layer.
         * 
         * @param values Values to assign to the nodes of the layer. Must be the same
         * size as the layer.
         */
        void load(const Real *const p_values);
        void feedforward();
        /**
         * @brief Back propagates through the layers, generating derivatives for
         * all inputs along the way.
         * 
         * This method kicks it off with the scores, which will have partial derivates
         * taken in respect to the inputs of the network.
         * 
         * @param p_scores Scores of each output node.
         */
        void back_propagate(const NetworkScore& p_scores);
        void train(const uint p_samples, const Real p_coef);
        void reset_training();

        /**
         * @brief Randomize the weights in the layer.
         * 
         * Nothing happens if this is the last layer. Will fail if
         * max is less than min.
         * 
         * @param min Minimum a weight can be.
         * @param max Maximum a weight can be.
         */
        void randomize_weights(const Real min, const Real max);

        void randomize_biases(const Real min, const Real max);

        inline const Real* const get_node_values() const {
            return this->m_nodes->get_ref();
        }

        inline std::shared_ptr<Layer> get_next() const {
            return this->m_next;
        }

        /**
         * @brief Set the next layer.
         *
         * @param next Next layer.
         */
        inline void set_next(std::shared_ptr<Layer> next) {
            this->m_next = next;
        }

        inline std::shared_ptr<Layer> get_prev() const {
            return this->m_prev;
        }

        /**
         * @brief Set the Prev object
         *
         * @param prev
         */
        inline void set_prev(std::shared_ptr<Layer> prev) {
            this->m_prev = prev;
        }

        inline ushort get_num_nodes() const {
            return this->m_nodes->get_num_rows();
        }

        inline const std::shared_ptr<const Matrix> get_nodes() const {
            return this->m_nodes;
        }

        inline const std::shared_ptr<const Matrix> get_weights() const {
            return this->m_weights;
        }

        inline const std::shared_ptr<const Matrix> get_biases() const {
            return this->m_biases;
        }

        inline const std::shared_ptr<const Matrix> get_d_weights() const {
            return this->m_dWeights;
        }

        inline const std::shared_ptr<const Matrix> get_d_biases() const {
            return this->m_dBiases;
        }

        inline const std::shared_ptr<const Matrix> get_d_weights_accum() const {
            return this->m_dWeightsAccum;
        }

        inline const std::shared_ptr<const Matrix> get_d_biases_accum() const {
            return this->m_dBiasesAccum;
        }

        inline const uint get_layer_num() const {
            return this->m_layerNum;
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
         * @param desc Descriptor of the network.
         */
        Network(const NetworkDescriptor m_desc);
        Network(const Network &_) = delete;
        ~Network();

        void randomize(const Real p_minBias, const Real p_maxBias, const Real p_minWeight, const Real p_maxWeight);
        void load(const Real* const p_input);
        std::vector<Real> feedforward();
        void back_propagate(const NetworkScore& p_scores);
        NetworkScore score(const Real* const p_expected) const;
        void train(const uint p_samples, const Real p_coef);
        void reset_training();

        std::string str(
            bool input,
            bool hidden,
            bool weights,
            bool biases,
            bool output,
            bool derivs,
            bool derivAccums
        ) const;
    };
}
