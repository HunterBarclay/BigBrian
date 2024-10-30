#include "brian/nn.h"

#define E 2.71828

namespace bb {
    namespace activation {
        Real leaky_re_lu(Real a) {
            return std::max(0.05 * a, a);
        }
        Real d_leaky_re_lu(Real a) {
            return a < 0 ? 0.05 : 1;
        }

        Real linear(Real a) {
            return a;
        }
        Real d_linear(Real a) {
            return 1;
        }

        Real sigmoid(Real a) {
            return 1.0 / (1.0 + powl(E, -a));
        }
        Real d_sigmoid(Real a) {
            return Sigmoid(a) * (1.0 - Sigmoid(a));
        }
    }
}