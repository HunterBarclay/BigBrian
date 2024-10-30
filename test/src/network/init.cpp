/**
 * @brief Test basic constructor usage. 
 */

#include <brian/prelude.h>
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int main(int argc, char** argv) {

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

    auto mat = std::make_unique<bb::Network>(desc);
    return 0;
}
