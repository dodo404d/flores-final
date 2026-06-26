#pragma once

#include <vector>

namespace flowercnn {

class SoftmaxCrossEntropy {
private:
    std::vector<double> probabilities_;
    int lastTarget_ = -1;

public:
    double forward(const std::vector<double>& logits, int targetLabel);
    std::vector<double> backward() const;
    const std::vector<double>& probabilities() const;
};

}
