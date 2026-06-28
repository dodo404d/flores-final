#pragma once

#include "core/Tensor.h"
#include <vector>
#include <cstddef>
#include <iosfwd>

namespace flowercnn {

class Conv2D {
private:
    std::size_t inputChannels_;
    std::size_t outputChannels_;
    std::size_t kernelSize_;
    std::size_t stride_;
    std::size_t padding_;

    /*
        Pesos organizados como:

        weights_[f][c][ky][kx]

        f  -> filtro o canal de salida
        c  -> canal de entrada
        ky -> fila del kernel
        kx -> columna del kernel
    */
    std::vector<double> weights_;
    std::vector<double> bias_;

    std::vector<double> gradWeights_;
    std::vector<double> gradBias_;

    Tensor cachedInput_;

    std::size_t weightIndex(
        std::size_t f,
        std::size_t c,
        std::size_t ky,
        std::size_t kx
    ) const;

public:
    Conv2D(
        std::size_t inputChannels,
        std::size_t outputChannels,
        std::size_t kernelSize,
        std::size_t stride = 1,
        std::size_t padding = 0
    );

    Tensor forward(const Tensor& input);

    Tensor backward(const Tensor& gradOutput);

    void applyGradients(double learningRate);

    void save(std::ostream& out) const;
    void load(std::istream& in);

    std::size_t inputChannels() const;
    std::size_t outputChannels() const;
    std::size_t kernelSize() const;
    std::size_t stride() const;
    std::size_t padding() const;
};

}