#include "training/OptimizerSGD.h"

namespace flowercnn {

OptimizerSGD::OptimizerSGD(double learningRate) : learningRate_(learningRate) {}

void OptimizerSGD::step(Dense& layer) const {
    layer.applyGradients(learningRate_);
}

double OptimizerSGD::learningRate() const {
    return learningRate_;
}

}
