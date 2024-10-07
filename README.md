# Big Brian
*"Sometimes my genius, it's... it's almost frightening." - Jeremy Clarkson*

When I initially created this project, I actually just misspelled "brain", so yeah.

This project is a playground for me to learn more about neural networks and machine learning.

## Goals
- [x] Convolutional neural network with hidden layers and per-node biases.
- [ ] Propagation of a number of a population of networks with a set of data, plus scoring and ranking for the population.
- [ ] Network training using back propagation.

## Usage

### Requirements
This project uses CMake version `^3.20` (literally just picked a version) and `C++14`.

### Building

To build, run the following commands:
```bash
# Create a build directory
mkdir build && cd build

# CMake to generate build files
cmake ../

# Build the project (you can replace this with whatever process is needed for the generator you used)
cmake --build .
```

*My default on my PC was set to "Visual Studio 17 2022" and it failed for some unknown reason when using the build command, but I normally do "MinGW Makefiles", and that worked so I don't know what's up with that.*

### Testing
There are unit tests setup with CTest, which can be run with the following:

```bash
ctest
```

### Running
Currently, there is no main program to use.

## Previous Iteration
This project was originally written in C#, implementing an convolutional neural network, utilizing back propagation for training.

It was able to "read" hand-written numbers from the MNIST database.

## License
Copyright (c) 2024 Hunter Barclay

[MIT License](/LICENSE.md).
