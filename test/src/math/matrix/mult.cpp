#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int test_src_math_matrix_mult(int argc, char** argv) {
    bb::Real a[] = {
        1,
        2
    };
    bb::Real b[] = {
        3, 4
    };

    bb::Real ab[] = {
        3, 4,
        6, 8
    };
    bb::Real ba[] = {
        11
    };

    auto matA = std::make_unique<bb::Matrix>(2, 1, a);
    ValidateMatrix(*matA, 2, 1, a);
    auto matB = std::make_unique<bb::Matrix>(1, 2, b);
    ValidateMatrix(*matB, 1, 2, b);
    auto matAB = matA->Mult(*matB);
    ValidateMatrix(*matAB, 2, 2, ab);
    auto matBA = matB->Mult(*matA);
    ValidateMatrix(*matBA, 1, 1, ba);

    return 0;
}