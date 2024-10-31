/**
 * @brief Test Feedforward. 
 */

#include <brian/prelude.h>
#include "util.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

#include <sstream>

void U_DSCSV_TestA() {
    auto data = std::string(
        "i0,i1,o0\n"
        "1,1,0\n"
        "1,0,1\n"
    );
    
    std::istringstream stream(data);
    auto dataOpt = bb::parse_deterministic_samples(stream);
    return;
    assert(dataOpt.has());
    auto dataParsed = dataOpt.value();
    printf("Columns parsed: %d\n", dataParsed.size());
    assert(dataParsed.size() == 2);
    return;
    auto row1 = &dataParsed.at(0);
    assert(bb::repsilon(row1->input()[0], 1));
    assert(bb::repsilon(row1->input()[1], 1));
    assert(bb::repsilon(row1->output()[0], 0));
    auto row2 = &dataParsed.at(1);
    assert(bb::repsilon(row2->input()[0], 1));
    assert(bb::repsilon(row2->input()[1], 0));
    assert(bb::repsilon(row2->output()[0], 1));
}

void U_DSCSV_TestB() {
    auto data = std::string(
        "i0,i1,o0\n"
        "2,5.0,0\n"
        "0,-34.1,1\n"
    );
    
    std::istringstream stream(data);
    auto dataOpt = bb::parse_deterministic_samples(stream);
    assert(dataOpt.has());
    auto dataParsed = dataOpt.value();
    assert(dataParsed.size() == 2);
    auto row1 = &dataParsed.at(0);
    assert(bb::repsilon(row1->input()[0], 2));
    assert(bb::repsilon(row1->input()[1], 5));
    assert(bb::repsilon(row1->output()[0], 0));
    auto row2 = &dataParsed.at(1);
    assert(bb::repsilon(row2->input()[0], 0));
    assert(bb::repsilon(row2->input()[1], -34.1));
    assert(bb::repsilon(row2->output()[0], 1));
}

int main(int argc, char** argv) {

    U_DSCSV_TestA();
    U_DSCSV_TestB();

    return 0;
}
