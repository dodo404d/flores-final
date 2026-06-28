#pragma once

#include <vector>
#include "core/Tensor.h"

namespace flowercnn {

class ReLU {
private:
    std::vector<double> cachedInput_;
    Tensor cachedTensorInput_;

public:
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradOutput) const;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput) const;
};

}