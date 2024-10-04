#include "math/Matrix.h"

#include <cstdio>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    bb::Real a[] = {
        1,
        2
    };
    bb::Real b[] = {
        3, 4
    };
    auto matA = std::make_unique<bb::Matrix>(2, 1, a);
    auto matB = std::make_unique<bb::Matrix>(1, 2, b);
    auto matAB = matA->Mult(*matB);
    auto matBA = matB->Mult(*matA);

    assert(matA->getRows() == 2);
    assert(matA->getCols() == 1);
    assert(matB->getRows() == 1);
    assert(matB->getCols() == 2);
    assert(matAB->getRows() == 2);
    assert(matAB->getCols() == 2);
    assert(matBA->getRows() == 1);
    assert(matBA->getCols() == 1);

    assert(matA->get(0, 0) == 1);
    assert(matA->get(1, 0) == 2);
    assert(matB->get(0, 0) == 3);
    assert(matB->get(0, 1) == 4);
    assert(matAB->get(0, 0) == 3);
    assert(matAB->get(0, 1) == 4);
    assert(matAB->get(1, 0) == 6);
    assert(matAB->get(1, 1) == 8);
    assert(matBA->get(0, 0) == 11);

    return 0;
}