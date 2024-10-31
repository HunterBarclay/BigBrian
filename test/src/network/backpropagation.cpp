/**
 * @brief Test backpropagation.
 * 
 * Just to verify no segfaults or anything.
 */

#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>
#include <vector>

void N_BP_TestA() {
    ushort layers[] = {
        3, 4, 5, 2
    };
    bb::NetworkDescriptor desc = {
        4,
        layers,
        bb::activation::leaky_re_lu,
        bb::activation::d_leaky_re_lu,
        bb::activation::linear,
        bb::activation::d_linear
    };
    bb::Real input[] = {
        1,
        2,
        3
    };
    bb::Real expected[] = {
        0,
        0
    };

    auto network = std::make_unique<bb::Network>(desc);
    network->randomize(0, 0, -1, -1);
    network->load(input);
    auto out = network->feedforward();

    bb::NetworkScore scores = network->score(expected);
    std::cout << "\t[ SCORE ]\n";
    for (auto iter = scores.nodeScores.begin(); iter != scores.nodeScores.end(); ++iter) {
        printf("%5.3g,\n", *iter);
    }
    network->back_propagate(scores);
    network->train(1, 0.1);
    std::cout << network->str(true, true, true, true, true, true, true);
    
    network->reset_training();
    std::cout << "\n\n[ ITERATION 1 ]\n" << network->str(true, true, true, true, true, true, true);
}

int main(int argc, char** argv) {

    N_BP_TestA();

    return 0;
}