#include <brian/prelude.h>
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int test_src_math_matrix_mutate(int argc, char** argv) {
    bb::Real data[] = {
        1, 2,
        3, 4,
        5, 6
    };

    bb::Real a[] = {
        4, 5,
        6, 7,
        8, 9
    };
    bb::Real b[] = {
         1,  4,
         9, 16,
        25, 36
    };

    auto matA = std::make_unique<bb::Matrix>(3, 2, data);
    matA->Mutate([](bb::Real r) -> bb::Real {
        return r + 3;
    });
    std::cout << matA->str();
    ValidateMatrix(*matA, 3, 2, a);

    auto matB = std::make_unique<bb::Matrix>(3, 2, data);
    matB->Mutate([](bb::Real r) -> bb::Real {
        return r * r;
    });
    std::cout << matA->str();
    ValidateMatrix(*matB, 3, 2, b);

    return 0;
}