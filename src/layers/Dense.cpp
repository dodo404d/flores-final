#include "layers/Dense.h"
#include "core/RandomInitializer.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>

namespace flowercnn {

Dense::Dense(std::size_t inputSize, std::size_t outputSize)
    : weights_(RandomInitializer::xavierMatrix(outputSize, inputSize)),
      bias_(outputSize, 0.0),
      gradWeights_(outputSize, inputSize, 0.0),
      gradBias_(outputSize, 0.0) {}

std::vector<double> Dense::forward(const std::vector<double>& input) {
    if (input.size() != weights_.cols()) {
        throw std::invalid_argument("Dense::forward recibio dimension incompatible");
    }

    cachedInput_ = input;
    std::vector<double> output = weights_.multiply(input);
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] += bias_[i];
    }
    return output;
}

std::vector<double> Dense::backward(const std::vector<double>& gradOutput) {
    if (gradOutput.size() != weights_.rows()) {
        throw std::invalid_argument("Dense::backward recibio dimension incompatible");
    }

    gradWeights_.fill(0.0);
    std::fill(gradBias_.begin(), gradBias_.end(), 0.0);

    for (std::size_t i = 0; i < weights_.rows(); ++i) {
        gradBias_[i] = gradOutput[i];
        for (std::size_t j = 0; j < weights_.cols(); ++j) {
            gradWeights_.at(i, j) = gradOutput[i] * cachedInput_[j];
        }
    }

    return weights_.transposeMultiply(gradOutput);
}

void Dense::applyGradients(double learningRate) {
    for (std::size_t i = 0; i < weights_.rows(); ++i) {
        for (std::size_t j = 0; j < weights_.cols(); ++j) {
            weights_.at(i, j) -= learningRate * gradWeights_.at(i, j);
        }
        bias_[i] -= learningRate * gradBias_[i];
    }
}

void Dense::save(std::ostream& out) const {
    out << "Dense " << inputSize() << " " << outputSize() << "\n";

    out << std::setprecision(17);

    out << weights_.size() << "\n";
    for (double value : weights_.data()) {
        out << value << " ";
    }
    out << "\n";

    out << bias_.size() << "\n";
    for (double value : bias_) {
        out << value << " ";
    }
    out << "\n";
}

void Dense::load(std::istream& in) {
    std::string tag;
    std::size_t inputSizeFile;
    std::size_t outputSizeFile;

    in >> tag >> inputSizeFile >> outputSizeFile;

    if (tag != "Dense") {
        throw std::runtime_error("Dense::load esperaba etiqueta Dense");
    }

    if (inputSizeFile != inputSize() || outputSizeFile != outputSize()) {
        throw std::runtime_error("Dense::load dimensiones incompatibles");
    }

    std::size_t weightsSize;
    in >> weightsSize;

    if (weightsSize != weights_.size()) {
        throw std::runtime_error("Dense::load tamano de pesos incompatible");
    }

    for (double& value : weights_.data()) {
        in >> value;
    }

    std::size_t biasSize;
    in >> biasSize;

    if (biasSize != bias_.size()) {
        throw std::runtime_error("Dense::load tamano de bias incompatible");
    }

    for (double& value : bias_) {
        in >> value;
    }
}

std::size_t Dense::inputSize() const { return weights_.cols(); }
std::size_t Dense::outputSize() const { return weights_.rows(); }

}
