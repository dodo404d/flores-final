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

Tensor ReLU::forward(const Tensor& input) {
    cachedTensorInput_ = input;

    Tensor output(input.height(), input.width(), input.channels(), 0.0);

    for (std::size_t y = 0; y < input.height(); ++y) {
        for (std::size_t x = 0; x < input.width(); ++x) {
            for (std::size_t c = 0; c < input.channels(); ++c) {
                output.at(y, x, c) = std::max(0.0, input.at(y, x, c));
            }
        }
    }

    return output;
}

Tensor ReLU::backward(const Tensor& gradOutput) const {
    if (gradOutput.height() != cachedTensorInput_.height() ||
        gradOutput.width() != cachedTensorInput_.width() ||
        gradOutput.channels() != cachedTensorInput_.channels()) {
        throw std::invalid_argument("ReLU::backward Tensor recibio dimension incompatible");
    }

    Tensor gradInput(
        gradOutput.height(),
        gradOutput.width(),
        gradOutput.channels(),
        0.0
    );

    for (std::size_t y = 0; y < gradOutput.height(); ++y) {
        for (std::size_t x = 0; x < gradOutput.width(); ++x) {
            for (std::size_t c = 0; c < gradOutput.channels(); ++c) {
                gradInput.at(y, x, c) =
                    cachedTensorInput_.at(y, x, c) > 0.0
                    ? gradOutput.at(y, x, c)
                    : 0.0;
            }
        }
    }

    return gradInput;
}

}