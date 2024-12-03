#pragma once

#include "brian/prelude.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>
#include <math.h>

void validate_matrix(const bb::Matrix& mat, const uint rows, const uint cols);
void validate_matrix(const bb::Matrix& mat, const uint rows, const uint cols, const bb::Real* const data);
