#pragma once

#include <iostream>

class NN {
private:
    int value;
public:
    NN(int value);
    ~NN();

    inline int getValue();
};