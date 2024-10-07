#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int test_src_math_matrix_add_reuse(int argc, char** argv) {
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

    bb::Real cPre[] = {
        -1, -1,
        -1, -1,
        -1, -1
    };
    bb::Real cPost[] = {
         8, 10,
        12, 14,
        16, 18
    };

    auto matA = std::make_unique<bb::Matrix>(3, 2, a);
    ValidateMatrix(*matA, 3, 2, a);
    auto matB = std::make_unique<bb::Matrix>(3, 2, b);
    ValidateMatrix(*matB, 3, 2, b);
    auto matC = std::make_unique<bb::Matrix>(3, 2, cPre);
    ValidateMatrix(*matC, 3, 2, cPre);

    matA->Add(*matB, *matC);
    std::cout << matC->str();
    ValidateMatrix(*matC, 3, 2, cPost);

    matA->Add(*matB, *matA);
    std::cout << matA->str();
    ValidateMatrix(*matA, 3, 2, cPost);

    return 0;
}