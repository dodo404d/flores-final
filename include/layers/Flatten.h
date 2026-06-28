#pragma once

#include "core/Tensor.h"
#include <vector>
#include <cstddef>

namespace flowercnn {

class Flatten {
private:
    std::size_t cachedHeight_;
    std::size_t cachedWidth_;
    std::size_t cachedChannels_;

public:
    Flatten();

    std::vector<double> forward(const Tensor& input);

    Tensor backward(const std::vector<double>& gradOutput) const;

    std::size_t outputSize() const;
};

}