#pragma once

#include <vector>
#include "core/Matrix.h"

namespace flowercnn {

class Dense {
private:
    Matrix weights_;
    std::vector<double> bias_;
    Matrix gradWeights_;
    std::vector<double> gradBias_;
    std::vector<double> cachedInput_;

public:
    Dense(std::size_t inputSize, std::size_t outputSize);

    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradOutput);
    void applyGradients(double learningRate);

    std::size_t inputSize() const;
    std::size_t outputSize() const;
};

}
