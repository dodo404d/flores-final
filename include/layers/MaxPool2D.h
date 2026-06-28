#pragma once

#include "core/Tensor.h"
#include <vector>
#include <cstddef>

namespace flowercnn {

class MaxPool2D {
private:
    std::size_t poolSize_;
    std::size_t stride_;

    Tensor cachedInput_;

    /*
        Guardamos la posicion del maximo para cada valor de salida.
        Esto todavia no se usa en forward, pero sera necesario para backward.
    */
    std::vector<std::size_t> maxY_;
    std::vector<std::size_t> maxX_;

    std::size_t index(
        std::size_t y,
        std::size_t x,
        std::size_t c,
        std::size_t outputWidth,
        std::size_t channels
    ) const;

public:
    MaxPool2D(
        std::size_t poolSize = 2,
        std::size_t stride = 2
    );

    Tensor forward(const Tensor& input);
    
    Tensor backward(const Tensor& gradOutput) const;

    std::size_t poolSize() const;
    std::size_t stride() const;
};

}