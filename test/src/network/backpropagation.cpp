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
        bb::LeakyReLU,
        bb::dLeakyReLU,
        bb::Linear,
        bb::dLinear
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
    network->Randomize(0, 0, -1, -1);
    network->Load(input);
    auto out = network->Feedforward();

    bb::NetworkScore scores = network->Score(expected);
    std::cout << "\t[ SCORE ]\n";
    for (auto iter = scores.nodeScores.begin(); iter != scores.nodeScores.end(); ++iter) {
        printf("%5.3g,\n", *iter);
    }
    network->BackPropagate(scores);
    network->Train(0.1);
    std::cout << network->str(true, true, true, true, true, true, true);
    
    network->ResetTraining();
    std::cout << "\n\n[ ITERATION 1 ]\n" << network->str(true, true, true, true, true, true, true);
}

int main(int argc, char** argv) {

    N_BP_TestA();

    return 0;
}