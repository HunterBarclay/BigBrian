#include "brian/prelude.h"
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

int main(int argc, char** argv) {
    bb::Real a[] = {
        1,
        2
    };
    bb::Real b[] = {
        3, 4
    };

    bb::Real cPre[] = {
        1, 2,
        3, 4
    };
    bb::Real cPost[] = {
        3, 4,
        6, 8
    };

    auto matA = std::make_unique<bb::Matrix>(2, 1, a);
    validate_matrix(*matA, 2, 1, a);
    auto matB = std::make_unique<bb::Matrix>(1, 2, b);
    validate_matrix(*matB, 1, 2, b);
    auto matC = std::make_unique<bb::Matrix>(2, 2, cPre);
    validate_matrix(*matC, 2, 2, cPre);

    matA->mult(*matB, *matC);
    validate_matrix(*matC, 2, 2, cPost);

    return 0;
}