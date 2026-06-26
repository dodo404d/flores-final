#include "data/ImageProcessor.h"
#include <algorithm>
#include <cmath>

namespace flowercnn {

Tensor ImageProcessor::resizeNearest(const Tensor& input, std::size_t newHeight, std::size_t newWidth) {
    if (input.height() == 0 || input.width() == 0 || input.channels() == 0) {
        return Tensor();
    }

    Tensor output(newHeight, newWidth, input.channels(), 0.0);

    for (std::size_t y = 0; y < newHeight; ++y) {
        std::size_t srcY = static_cast<std::size_t>(
            std::floor((static_cast<double>(y) / static_cast<double>(newHeight)) * input.height())
        );
        srcY = std::min(srcY, input.height() - 1);

        for (std::size_t x = 0; x < newWidth; ++x) {
            std::size_t srcX = static_cast<std::size_t>(
                std::floor((static_cast<double>(x) / static_cast<double>(newWidth)) * input.width())
            );
            srcX = std::min(srcX, input.width() - 1);

            for (std::size_t c = 0; c < input.channels(); ++c) {
                output.at(y, x, c) = input.at(srcY, srcX, c);
            }
        }
    }
    return output;
}

Tensor ImageProcessor::ensureRange01(const Tensor& input) {
    Tensor output(input.height(), input.width(), input.channels(), 0.0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output.data()[i] = std::clamp(input.data()[i], 0.0, 1.0);
    }
    return output;
}

}
