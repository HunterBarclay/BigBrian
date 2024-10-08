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
    auto mat = std::make_unique<bb::Matrix>(1, 2);
    ValidateMatrix(*mat, 1, 2);
    return 0;
}
