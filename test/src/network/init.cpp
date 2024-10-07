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

int test_src_network_init(int argc, char** argv) {

    ushort layers[] = {
        1, 2, 1
    };
    bb::NetworkDescriptor desc = {
        3,
        layers,
        bb::LeakyReLU,
        bb::Linear
    };

    auto mat = std::make_unique<bb::Network>(desc);
    return 0;
}
