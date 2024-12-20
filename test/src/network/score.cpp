/**
 * @brief Test scoring. 
 */

#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>
#include <vector>

void N_S_TestA() {
    ushort test = 0x1234;
    ushort layers[] = {
        3, 2
    };
    bb::NetworkDescriptor desc = {
        2,
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
        -6,
        -6
    };

    auto network = std::make_unique<bb::Network>(desc);
    network->randomize(0, 0, -1, -1);
    network->load(input);
    auto out = network->feedforward();

    std::cout << network->str(true, true, true, true, true, false, false);

    bb::NetworkScore score = network->score(expected);
    
    for (auto iter = score.nodeScores.begin(); iter != score.nodeScores.end(); ++iter) {
        assert(bb::repsilon(*iter, 0.0));
    }
    assert(bb::repsilon(score.overallScore, 0.0));
}

void N_S_TestB() {
    ushort layers[] = {
        3, 2
    };
    bb::NetworkDescriptor desc = {
        2,
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
         3,
        -3
    };
    bb::Real expectedScores[] = {
        81,
        9
    };
    bb::Real expectedOverall = 90;

    auto network = std::make_unique<bb::Network>(desc);
    network->randomize(0, 0, -1, -1);
    network->load(input);
    auto out = network->feedforward();

    std::cout << network->str(true, true, true, true, true, false, false);

    bb::NetworkScore score = network->score(expected);
    std::cout << "\t[ SCORE ]\n";
    for (auto iter = score.nodeScores.begin(); iter != score.nodeScores.end(); ++iter) {
        printf("%5.3g,\n", *iter);
    }
    printf("\tOverall: %5.3g\n", score.overallScore);
    
    for (ushort i = 0; i < score.nodeScores.size(); ++i) {
        assert(bb::repsilon(score.nodeScores.at(i), expectedScores[i]));
    }
    assert(bb::repsilon(score.overallScore, expectedOverall));
}

int main(int argc, char** argv) {

    N_S_TestA();
    N_S_TestB();

    return 0;
}