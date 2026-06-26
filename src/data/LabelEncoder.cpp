#include "data/LabelEncoder.h"
#include <stdexcept>

namespace flowercnn {

LabelEncoder::LabelEncoder()
    : LabelEncoder({"daisy", "dandelion", "rose", "sunflower", "tulip"}) {}

LabelEncoder::LabelEncoder(const std::vector<std::string>& classNames) : classes_(classNames) {
    for (std::size_t i = 0; i < classes_.size(); ++i) {
        nameToId_[classes_[i]] = static_cast<int>(i);
    }
}

int LabelEncoder::encode(const std::string& className) const {
    auto it = nameToId_.find(className);
    if (it == nameToId_.end()) {
        throw std::invalid_argument("Clase desconocida: " + className);
    }
    return it->second;
}

std::string LabelEncoder::decode(int label) const {
    if (label < 0 || label >= static_cast<int>(classes_.size())) {
        throw std::invalid_argument("Etiqueta fuera de rango");
    }
    return classes_[static_cast<std::size_t>(label)];
}

const std::vector<std::string>& LabelEncoder::classes() const {
    return classes_;
}

std::size_t LabelEncoder::size() const {
    return classes_.size();
}

}
