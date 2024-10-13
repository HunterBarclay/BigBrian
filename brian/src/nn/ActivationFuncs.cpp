#include "brian/NN.h"

#define E 2.71828

namespace bb {
    Real LeakyReLU(Real a) {
        return std::max(0.05 * a, a);
    }
    Real dLeakyReLU(Real a) {
        return a < 0 ? 0.05 : 1;
    }

    Real Linear(Real a) {
        return a;
    }
    Real dLinear(Real a) {
        return 1;
    }

    Real Sigmoid(Real a) {
        return 1.0 / (1.0 + powl(E, -a));
    }
    Real dSigmoid(Real a) {
        return Sigmoid(a) * (1.0 - Sigmoid(a));
    }
}