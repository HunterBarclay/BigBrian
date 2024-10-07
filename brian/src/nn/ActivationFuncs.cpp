#include "brian/NN.h"

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
}