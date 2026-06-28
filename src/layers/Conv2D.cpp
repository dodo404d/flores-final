#include "layers/Conv2D.h"
#include "core/RandomInitializer.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>

namespace flowercnn {

Conv2D::Conv2D(
    std::size_t inputChannels,
    std::size_t outputChannels,
    std::size_t kernelSize,
    std::size_t stride,
    std::size_t padding
)
    : inputChannels_(inputChannels),
      outputChannels_(outputChannels),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      weights_(outputChannels * inputChannels * kernelSize * kernelSize, 0.0),
      bias_(outputChannels, 0.0),
      gradWeights_(outputChannels * inputChannels * kernelSize * kernelSize, 0.0),
      gradBias_(outputChannels, 0.0) {

    if (inputChannels_ == 0 || outputChannels_ == 0 || kernelSize_ == 0 || stride_ == 0) {
        throw std::invalid_argument("Conv2D recibio parametros invalidos");
    }

    double fanIn = static_cast<double>(inputChannels_ * kernelSize_ * kernelSize_);
    double fanOut = static_cast<double>(outputChannels_ * kernelSize_ * kernelSize_);
    double limit = std::sqrt(6.0 / (fanIn + fanOut));

    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = RandomInitializer::uniform(-limit, limit);
    }
}

std::size_t Conv2D::weightIndex(
    std::size_t f,
    std::size_t c,
    std::size_t ky,
    std::size_t kx
) const {
    return ((f * inputChannels_ + c) * kernelSize_ + ky) * kernelSize_ + kx;
}

Tensor Conv2D::forward(const Tensor& input) {
    if (input.channels() != inputChannels_) {
        throw std::invalid_argument("Conv2D::forward recibio un numero de canales incompatible");
    }

    if (input.height() + 2 * padding_ < kernelSize_ ||
        input.width() + 2 * padding_ < kernelSize_) {
        throw std::invalid_argument("Conv2D::forward recibio imagen demasiado pequena para el kernel");
    }

    cachedInput_ = input;

    std::size_t outputHeight = ((input.height() + 2 * padding_ - kernelSize_) / stride_) + 1;
    std::size_t outputWidth = ((input.width() + 2 * padding_ - kernelSize_) / stride_) + 1;

    Tensor output(outputHeight, outputWidth, outputChannels_, 0.0);

    for (std::size_t f = 0; f < outputChannels_; ++f) {
        for (std::size_t oy = 0; oy < outputHeight; ++oy) {
            for (std::size_t ox = 0; ox < outputWidth; ++ox) {

                double sum = bias_[f];

                for (std::size_t c = 0; c < inputChannels_; ++c) {
                    for (std::size_t ky = 0; ky < kernelSize_; ++ky) {
                        for (std::size_t kx = 0; kx < kernelSize_; ++kx) {

                            int inputY = static_cast<int>(oy * stride_ + ky) - static_cast<int>(padding_);
                            int inputX = static_cast<int>(ox * stride_ + kx) - static_cast<int>(padding_);

                            if (inputY >= 0 && inputY < static_cast<int>(input.height()) &&
                                inputX >= 0 && inputX < static_cast<int>(input.width())) {

                                double pixel = input.at(
                                    static_cast<std::size_t>(inputY),
                                    static_cast<std::size_t>(inputX),
                                    c
                                );

                                double weight = weights_[weightIndex(f, c, ky, kx)];

                                sum += pixel * weight;
                            }
                        }
                    }
                }

                output.at(oy, ox, f) = sum;
            }
        }
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& gradOutput) {
    if (cachedInput_.height() == 0 ||
        cachedInput_.width() == 0 ||
        cachedInput_.channels() == 0) {
        throw std::runtime_error("Conv2D::backward llamado antes de forward");
    }

    std::size_t expectedHeight = ((cachedInput_.height() + 2 * padding_ - kernelSize_) / stride_) + 1;
    std::size_t expectedWidth = ((cachedInput_.width() + 2 * padding_ - kernelSize_) / stride_) + 1;

    if (gradOutput.height() != expectedHeight ||
        gradOutput.width() != expectedWidth ||
        gradOutput.channels() != outputChannels_) {
        throw std::invalid_argument("Conv2D::backward recibio dimensiones incompatibles");
    }

    std::fill(gradWeights_.begin(), gradWeights_.end(), 0.0);
    std::fill(gradBias_.begin(), gradBias_.end(), 0.0);

    Tensor gradInput(
        cachedInput_.height(),
        cachedInput_.width(),
        cachedInput_.channels(),
        0.0
    );

    for (std::size_t f = 0; f < outputChannels_; ++f) {
        for (std::size_t oy = 0; oy < gradOutput.height(); ++oy) {
            for (std::size_t ox = 0; ox < gradOutput.width(); ++ox) {

                double grad = gradOutput.at(oy, ox, f);
                gradBias_[f] += grad;

                for (std::size_t c = 0; c < inputChannels_; ++c) {
                    for (std::size_t ky = 0; ky < kernelSize_; ++ky) {
                        for (std::size_t kx = 0; kx < kernelSize_; ++kx) {

                            int inputY = static_cast<int>(oy * stride_ + ky) - static_cast<int>(padding_);
                            int inputX = static_cast<int>(ox * stride_ + kx) - static_cast<int>(padding_);

                            if (inputY >= 0 && inputY < static_cast<int>(cachedInput_.height()) &&
                                inputX >= 0 && inputX < static_cast<int>(cachedInput_.width())) {

                                std::size_t iy = static_cast<std::size_t>(inputY);
                                std::size_t ix = static_cast<std::size_t>(inputX);
                                std::size_t wi = weightIndex(f, c, ky, kx);

                                gradWeights_[wi] += grad * cachedInput_.at(iy, ix, c);
                                gradInput.at(iy, ix, c) += grad * weights_[wi];
                            }
                        }
                    }
                }
            }
        }
    }

    return gradInput;
}

void Conv2D::applyGradients(double learningRate) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learningRate * gradWeights_[i];
    }

    for (std::size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learningRate * gradBias_[i];
    }
}

void Conv2D::save(std::ostream& out) const {
    out << "Conv2D "
        << inputChannels_ << " "
        << outputChannels_ << " "
        << kernelSize_ << " "
        << stride_ << " "
        << padding_ << "\n";

    out << std::setprecision(17);

    out << weights_.size() << "\n";
    for (double value : weights_) {
        out << value << " ";
    }
    out << "\n";

    out << bias_.size() << "\n";
    for (double value : bias_) {
        out << value << " ";
    }
    out << "\n";
}

void Conv2D::load(std::istream& in) {
    std::string tag;
    std::size_t inputChannelsFile;
    std::size_t outputChannelsFile;
    std::size_t kernelSizeFile;
    std::size_t strideFile;
    std::size_t paddingFile;

    in >> tag
       >> inputChannelsFile
       >> outputChannelsFile
       >> kernelSizeFile
       >> strideFile
       >> paddingFile;

    if (tag != "Conv2D") {
        throw std::runtime_error("Conv2D::load esperaba etiqueta Conv2D");
    }

    if (inputChannelsFile != inputChannels_ ||
        outputChannelsFile != outputChannels_ ||
        kernelSizeFile != kernelSize_ ||
        strideFile != stride_ ||
        paddingFile != padding_) {
        throw std::runtime_error("Conv2D::load dimensiones incompatibles");
    }

    std::size_t weightsSize;
    in >> weightsSize;

    if (weightsSize != weights_.size()) {
        throw std::runtime_error("Conv2D::load tamano de pesos incompatible");
    }

    for (double& value : weights_) {
        in >> value;
    }

    std::size_t biasSize;
    in >> biasSize;

    if (biasSize != bias_.size()) {
        throw std::runtime_error("Conv2D::load tamano de bias incompatible");
    }

    for (double& value : bias_) {
        in >> value;
    }
}


std::size_t Conv2D::inputChannels() const {
    return inputChannels_;
}

std::size_t Conv2D::outputChannels() const {
    return outputChannels_;
}

std::size_t Conv2D::kernelSize() const {
    return kernelSize_;
}

std::size_t Conv2D::stride() const {
    return stride_;
}

std::size_t Conv2D::padding() const {
    return padding_;
}

}