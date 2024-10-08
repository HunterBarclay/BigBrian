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
    ValidateMatrix(*matA, 2, 1, a);
    auto matB = std::make_unique<bb::Matrix>(1, 2, b);
    ValidateMatrix(*matB, 1, 2, b);
    auto matC = std::make_unique<bb::Matrix>(2, 2, cPre);
    ValidateMatrix(*matC, 2, 2, cPre);

    matA->Mult(*matB, *matC);
    ValidateMatrix(*matC, 2, 2, cPost);

    return 0;
}