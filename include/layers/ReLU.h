#pragma once

#include <vector>

namespace flowercnn {

class ReLU {
private:
    std::vector<double> cachedInput_;

public:
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradOutput) const;
};

}
