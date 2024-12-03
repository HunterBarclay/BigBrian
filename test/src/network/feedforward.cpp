/**
 * @brief Test Feedforward. 
 */

#include <brian/prelude.h>
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>
#include <vector>

void N_FF_TestA() {
    ushort layers[] = {
        1, 2, 1
    };
    bb::NetworkDescriptor desc = {
        3,
        layers,
        bb::activation::leaky_re_lu,
        bb::activation::d_leaky_re_lu,
        bb::activation::linear,
        bb::activation::d_linear
    };
    bb::Real input[] = {
        5
    };
    bb::Real expected[] = {
        10
    };

    auto mat = std::make_unique<bb::Network>(desc);
    mat->randomize(0, 0, 1, 1);
    mat->load(input);
    auto out = mat->feedforward();

    std::cout << mat->str(true, true, true, true, true, false, false);
    
    for (size_t i = 0; i < out.size(); ++i) {
        assert(bb::repsilon(expected[i], out.at(i)));
    }
}

void N_FF_TestB() {
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
        -6, // TODO
        -6
    };

    auto mat = std::make_unique<bb::Network>(desc);
    mat->randomize(0, 0, -1, -1);
    mat->load(input);
    auto out = mat->feedforward();

    std::cout << mat->str(true, true, true, true, true, false, false);
    
    for (size_t i = 0; i < out.size(); ++i) {
        assert(bb::repsilon(expected[i], out.at(i)));
    }
}

int main(int argc, char** argv) {

    N_FF_TestA();
    N_FF_TestB();

    return 0;
}
