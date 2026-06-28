#include "layers/MaxPool2D.h"

#include <stdexcept>
#include <limits>

namespace flowercnn {

MaxPool2D::MaxPool2D(
    std::size_t poolSize,
    std::size_t stride
)
    : poolSize_(poolSize),
      stride_(stride) {

    if (poolSize_ == 0 || stride_ == 0) {
        throw std::invalid_argument("MaxPool2D recibio parametros invalidos");
    }
}

std::size_t MaxPool2D::index(
    std::size_t y,
    std::size_t x,
    std::size_t c,
    std::size_t outputWidth,
    std::size_t channels
) const {
    return (y * outputWidth + x) * channels + c;
}

Tensor MaxPool2D::forward(const Tensor& input) {
    if (input.height() < poolSize_ || input.width() < poolSize_) {
        throw std::invalid_argument("MaxPool2D::forward recibio un tensor demasiado pequeno");
    }

    cachedInput_ = input;

    std::size_t outputHeight = ((input.height() - poolSize_) / stride_) + 1;
    std::size_t outputWidth = ((input.width() - poolSize_) / stride_) + 1;
    std::size_t channels = input.channels();

    Tensor output(outputHeight, outputWidth, channels, 0.0);

    maxY_.assign(outputHeight * outputWidth * channels, 0);
    maxX_.assign(outputHeight * outputWidth * channels, 0);

    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t oy = 0; oy < outputHeight; ++oy) {
            for (std::size_t ox = 0; ox < outputWidth; ++ox) {

                std::size_t startY = oy * stride_;
                std::size_t startX = ox * stride_;

                double maxValue = -std::numeric_limits<double>::infinity();
                std::size_t bestY = startY;
                std::size_t bestX = startX;

                for (std::size_t py = 0; py < poolSize_; ++py) {
                    for (std::size_t px = 0; px < poolSize_; ++px) {
                        std::size_t inputY = startY + py;
                        std::size_t inputX = startX + px;

                        double value = input.at(inputY, inputX, c);

                        if (value > maxValue) {
                            maxValue = value;
                            bestY = inputY;
                            bestX = inputX;
                        }
                    }
                }

                output.at(oy, ox, c) = maxValue;

                std::size_t flatIndex = index(oy, ox, c, outputWidth, channels);
                maxY_[flatIndex] = bestY;
                maxX_[flatIndex] = bestX;
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(const Tensor& gradOutput) const {
    if (cachedInput_.height() == 0 || cachedInput_.width() == 0 || cachedInput_.channels() == 0) {
        throw std::runtime_error("MaxPool2D::backward llamado antes de forward");
    }

    std::size_t expectedHeight = ((cachedInput_.height() - poolSize_) / stride_) + 1;
    std::size_t expectedWidth = ((cachedInput_.width() - poolSize_) / stride_) + 1;
    std::size_t channels = cachedInput_.channels();

    if (gradOutput.height() != expectedHeight ||
        gradOutput.width() != expectedWidth ||
        gradOutput.channels() != channels) {
        throw std::invalid_argument("MaxPool2D::backward recibio dimensiones incompatibles");
    }

    Tensor gradInput(
        cachedInput_.height(),
        cachedInput_.width(),
        cachedInput_.channels(),
        0.0
    );

    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t oy = 0; oy < gradOutput.height(); ++oy) {
            for (std::size_t ox = 0; ox < gradOutput.width(); ++ox) {
                std::size_t flatIndex = index(oy, ox, c, gradOutput.width(), channels);

                std::size_t bestY = maxY_[flatIndex];
                std::size_t bestX = maxX_[flatIndex];

                gradInput.at(bestY, bestX, c) += gradOutput.at(oy, ox, c);
            }
        }
    }

    return gradInput;
}

std::size_t MaxPool2D::poolSize() const {
    return poolSize_;
}

std::size_t MaxPool2D::stride() const {
    return stride_;
}

}