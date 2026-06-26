#include "layers/ReLU.h"
#include <algorithm>
#include <stdexcept>

namespace flowercnn {

std::vector<double> ReLU::forward(const std::vector<double>& input) {
    cachedInput_ = input;
    std::vector<double> output(input.size(), 0.0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

std::vector<double> ReLU::backward(const std::vector<double>& gradOutput) const {
    if (gradOutput.size() != cachedInput_.size()) {
        throw std::invalid_argument("ReLU::backward recibio dimension incompatible");
    }

    std::vector<double> gradInput(gradOutput.size(), 0.0);
    for (std::size_t i = 0; i < gradOutput.size(); ++i) {
        gradInput[i] = cachedInput_[i] > 0.0 ? gradOutput[i] : 0.0;
    }
    return gradInput;
}

}
