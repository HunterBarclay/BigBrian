/**
 * @brief Test training to fit specific functions.
 */

#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>
#include <vector>

bb::Real i1[] = {
    0, 0
};
bb::Real o1[] = {
    0
};

bb::Real i2[] = {
    0, 1
};
bb::Real o2[] = {
    1
};

bb::Real i3[] = {
    1, 0
};
bb::Real o3[] = {
    1
};

bb::Real i4[] = {
    1, 1
};
bb::Real o4[] = {
    0
};

void Train_XOR() {
    ushort layers[] = {
        2, 3, 3, 1
    };
    bb::NetworkDescriptor desc = {
        4,
        layers,
        bb::LeakyReLU,
        bb::dLeakyReLU,
        bb::Linear,
        bb::dLinear
    };
    // bb::NetworkDescriptor desc = {
    //     3,
    //     layers,
    //     bb::Sigmoid,
    //     bb::dSigmoid,
    //     bb::Sigmoid,
    //     bb::dSigmoid
    // };

    bb::Population pop(1, desc);
    pop.PushSample({ input: i1, output: o1 });
    pop.PushSample({ input: i2, output: o2 });
    pop.PushSample({ input: i3, output: o3 });
    pop.PushSample({ input: i4, output: o4 });

    std::cout << "Initial Score: " << pop.getAverageScore() << "\n";
    for (uint i = 0; i < 100; ++i) {
        pop.Iterate(false);
        std::cout << "Score (" << pop.getNumIterations() << "): " << pop.getAverageScore() << "\n";
    }

    pop.Iterate(true);

    assert(0);
}

int main(int argc, char** argv) {

    Train_XOR();

    return 0;
}