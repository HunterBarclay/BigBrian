#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int main(int argc, char** argv) {
    bb::Real a[] = {
        1, 2,
        3, 4,
        5, 6
    };
    bb::Real b[] = {
         7,  8,
         9, 10,
        11, 12
    };

    bb::Real c[] = {
         8, 10,
        12, 14,
        16, 18
    };

    auto matA = std::make_unique<bb::Matrix>(3, 2, a);
    ValidateMatrix(*matA, 3, 2, a);
    auto matB = std::make_unique<bb::Matrix>(3, 2, b);
    ValidateMatrix(*matB, 3, 2, b);
    auto matAB = matA->Add(*matB);
    ValidateMatrix(*matAB, 3, 2, c);

    return 0;
}