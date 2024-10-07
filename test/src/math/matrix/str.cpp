#include <brian/prelude.h>
#include <iostream>
#include <cstring>
#include <cassert>

auto aOut = "Matrix 3 x 2 [\n"
"\t    1,     2, \n"
"\t    3,     4, \n"
"\t    5,     6, \n"
"]\n";

auto bOut = "Matrix 2 x 2 [\n"
"\t 1.25,   0.5, \n"
"\t   -4,  -5.6, \n"
"]\n";

int test_src_math_matrix_str(int argc, char** argv) {
    bb::Real a[] = {
        1, 2,
        3, 4,
        5, 6
    };
    auto aM = std::make_unique<bb::Matrix>(3, 2, a);
    auto aStr = aM->str();
    std::cout << "Expected:\n" << aOut;
    std::cout << "Actual:\n" << aStr;
    
    assert(strcmp(aOut, aStr.c_str()) == 0);

    bb::Real b[] = {
        1.25, 0.5,
        -4.0, -5.6,
    };
    auto bM = std::make_unique<bb::Matrix>(2, 2, b);
    auto bStr = bM->str();
    std::cout << "Expected:\n" << bOut;
    std::cout << "Actual:\n" << bStr;

    assert(strcmp(bOut, bStr.c_str()) == 0);

    return 0;
}