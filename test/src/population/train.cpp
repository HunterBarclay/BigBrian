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

#define MAX_ITERATIONS 100000
#define UPDATE_FREQ 1000
#define SCORE_ACCEPTANCE_THRESHOLD static_cast<bb::Real>(0.01)

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
    //     4,
    //     layers,
    //     bb::Sigmoid,
    //     bb::dSigmoid,
    //     bb::Linear,
    //     bb::dLinear
    // };

    bb::DeterministicPopulation pop(1, desc);
    pop.PushSample({ input: i1, output: o1 });
    pop.PushSample({ input: i2, output: o2 });
    pop.PushSample({ input: i3, output: o3 });
    pop.PushSample({ input: i4, output: o4 });

    std::cout << "Initial Score: " << pop.getAverageScore() << "\n";
    for (uint i = 0; i < MAX_ITERATIONS; ++i) {
        pop.Iterate(false);
        const uint iterations = pop.getNumIterations();
        const bb::Real score = pop.getAverageScore();
        if (iterations % UPDATE_FREQ == 0) {
            std::cout << "Score (" << iterations << "): " << score << "\n";
        }

        if (score <= SCORE_ACCEPTANCE_THRESHOLD) {
            printf("=====\n");
            printf("Acceptable score: %5.3g\n", score);
            printf("=====\n");
            break;
        }
    }

    pop.Iterate(true);

    assert(0);
}

int main(int argc, char** argv) {

    Train_XOR();

    return 0;
}