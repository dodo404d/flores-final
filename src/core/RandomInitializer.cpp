#include "core/RandomInitializer.h"
#include <cmath>

namespace flowercnn {

std::mt19937 RandomInitializer::engine_(42);

void RandomInitializer::setSeed(unsigned int seed) {
    engine_.seed(seed);
}

double RandomInitializer::uniform(double minValue, double maxValue) {
    std::uniform_real_distribution<double> dist(minValue, maxValue);
    return dist(engine_);
}

Matrix RandomInitializer::xavierMatrix(std::size_t rows, std::size_t cols) {
    double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
    Matrix matrix(rows, cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix.at(i, j) = uniform(-limit, limit);
        }
    }
    return matrix;
}

std::vector<double> RandomInitializer::zeros(std::size_t size) {
    return std::vector<double>(size, 0.0);
}

}
