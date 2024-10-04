#include "NN.h"

#include <iostream>

NN::NN(int value) : value(value) {
    std::cout << "NN: " << value << std::endl;
}

NN::~NN() {
    std::cout << "Bye.\n";
}

inline int NN::getValue() {
    return this->value;
}