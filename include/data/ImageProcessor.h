#pragma once

#include "core/Tensor.h"

namespace flowercnn {

class ImageProcessor {
public:
    static Tensor resizeNearest(const Tensor& input, std::size_t newHeight, std::size_t newWidth);
    static Tensor ensureRange01(const Tensor& input);
};

}
