#pragma once

#include <random>
#include <vector>
#include "core/Matrix.h"

namespace flowercnn {

class RandomInitializer {
private:
    static std::mt19937 engine_;

public:
    static void setSeed(unsigned int seed);
    static double uniform(double minValue, double maxValue);
    static Matrix xavierMatrix(std::size_t rows, std::size_t cols);
    static std::vector<double> zeros(std::size_t size);
};

}
