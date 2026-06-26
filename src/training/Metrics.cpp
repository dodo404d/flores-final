#include "training/Metrics.h"
#include <algorithm>
#include <stdexcept>

namespace flowercnn {

int Metrics::argmax(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Metrics::argmax recibio vector vacio");
    }
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

double Metrics::accuracy(const std::vector<int>& predictions, const std::vector<int>& labels) {
    if (predictions.size() != labels.size() || labels.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(labels.size());
}

}
