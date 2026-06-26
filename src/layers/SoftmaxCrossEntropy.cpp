#include "layers/SoftmaxCrossEntropy.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flowercnn {

double SoftmaxCrossEntropy::forward(const std::vector<double>& logits, int targetLabel) {
    if (logits.empty()) {
        throw std::invalid_argument("SoftmaxCrossEntropy::forward recibio logits vacios");
    }
    if (targetLabel < 0 || targetLabel >= static_cast<int>(logits.size())) {
        throw std::invalid_argument("Etiqueta fuera de rango en SoftmaxCrossEntropy");
    }

    lastTarget_ = targetLabel;
    probabilities_.assign(logits.size(), 0.0);

    double maxLogit = *std::max_element(logits.begin(), logits.end());
    double sumExp = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities_[i] = std::exp(logits[i] - maxLogit);
        sumExp += probabilities_[i];
    }

    for (double& p : probabilities_) {
        p /= sumExp;
    }

    const double epsilon = 1e-12;
    return -std::log(probabilities_[static_cast<std::size_t>(targetLabel)] + epsilon);
}

std::vector<double> SoftmaxCrossEntropy::backward() const {
    if (lastTarget_ < 0 || probabilities_.empty()) {
        throw std::runtime_error("SoftmaxCrossEntropy::backward llamado antes de forward");
    }

    std::vector<double> grad = probabilities_;
    grad[static_cast<std::size_t>(lastTarget_)] -= 1.0;
    return grad;
}

const std::vector<double>& SoftmaxCrossEntropy::probabilities() const {
    return probabilities_;
}

}
