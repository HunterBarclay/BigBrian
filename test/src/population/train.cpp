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

#define MAX_ITERATIONS 1000
#define UPDATE_FREQ 100
#define SCORE_ACCEPTANCE_THRESHOLD static_cast<bb::Real>(0.0001)

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
        bb::activation::leaky_re_lu,
        bb::activation::d_leaky_re_lu,
        bb::activation::linear,
        bb::activation::d_linear
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
    pop.push_sample(bb::DeterministicSample(2, 1, i1, o1));
    pop.push_sample(bb::DeterministicSample(2, 1, i2, o2));
    pop.push_sample(bb::DeterministicSample(2, 1, i3, o3));
    pop.push_sample(bb::DeterministicSample(2, 1, i4, o4));

    std::cout << "Initial Score: " << pop.get_average_score() << "\n";
    for (uint i = 0; i < MAX_ITERATIONS; ++i) {
        pop.iterate(false);
        const uint iterations = pop.get_num_iterations();
        const bb::Real score = pop.get_average_score();
        if (iterations % UPDATE_FREQ == 0) {
            std::cout << "Score (" << iterations << "): " << score << "\n";
        }

        if (score <= SCORE_ACCEPTANCE_THRESHOLD) {
            printf("=====\n");
            printf("Acceptable score: %5.3g\n", score);
            printf("(@ Iteration %d)\n", iterations);
            printf("=====\n");
            break;
        }
    }

    pop.iterate(true);
}

void Train_Circles() {
    ushort layers[] = {
        2, 8, 8, 5, 5, 1
    };
    bb::NetworkDescriptor desc = {
        6,
        layers,
        bb::activation::leaky_re_lu,
        bb::activation::d_leaky_re_lu,
        bb::activation::linear,
        bb::activation::d_linear
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
    bb::Real minX = -10;
    bb::Real minY = -10;
    bb::Real maxX = 10;
    bb::Real maxY = 10;
    bb::Real grain = 0.1;
    bb::Real rangeX = maxX - minX;
    bb::Real rangeY = maxY - minY;
    bb::Real r = 0.0;
    for (bb::Real x = minX; x <= maxX; x += rangeX * grain) {
        for (bb::Real y = minY; y <= maxY; y += rangeY * grain) {
            bb::Real i[2] = {
                x, y
            };
            bb::Real o[1] = {
                bb::rsqrt(x * x + y + y) < 4.0 ? 0.0 : 1.0
            };
            pop.push_sample(bb::DeterministicSample(2, 1, i, o));
        }
    }

    std::cout << "Initial Score: " << pop.get_average_score() << "\n";
    for (uint i = 0; i < MAX_ITERATIONS; ++i) {
        pop.iterate(false);
        const uint iterations = pop.get_num_iterations();
        const bb::Real score = pop.get_average_score();
        if (iterations % UPDATE_FREQ == 0) {
            std::cout << "Score (" << iterations << "): " << score << "\n";
        }

        if (score <= SCORE_ACCEPTANCE_THRESHOLD) {
            printf("=====\n");
            printf("Acceptable score: %5.3g\n", score);
            printf("(@ Iteration %d)\n", iterations);
            printf("=====\n");
            break;
        }
    }

    pop.iterate(true);

    assert(0);
}

int main(int argc, char** argv) {

    // Train_XOR();
    Train_Circles();

    return 0;
}