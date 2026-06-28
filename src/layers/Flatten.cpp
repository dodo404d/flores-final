#include "layers/Flatten.h"

#include <stdexcept>

namespace flowercnn {

Flatten::Flatten()
    : cachedHeight_(0),
      cachedWidth_(0),
      cachedChannels_(0) {}

std::vector<double> Flatten::forward(const Tensor& input) {
    cachedHeight_ = input.height();
    cachedWidth_ = input.width();
    cachedChannels_ = input.channels();

    return input.flatten();
}

Tensor Flatten::backward(const std::vector<double>& gradOutput) const {
    if (cachedHeight_ == 0 || cachedWidth_ == 0 || cachedChannels_ == 0) {
        throw std::runtime_error("Flatten::backward llamado antes de forward");
    }

    std::size_t expectedSize = cachedHeight_ * cachedWidth_ * cachedChannels_;

    if (gradOutput.size() != expectedSize) {
        throw std::invalid_argument("Flatten::backward recibio un gradiente con tamano incorrecto");
    }

    Tensor gradInput(cachedHeight_, cachedWidth_, cachedChannels_, 0.0);

    for (std::size_t i = 0; i < gradOutput.size(); ++i) {
        gradInput.data()[i] = gradOutput[i];
    }

    return gradInput;
}

std::size_t Flatten::outputSize() const {
    return cachedHeight_ * cachedWidth_ * cachedChannels_;
}

}