#pragma once

#include "layers/Dense.h"

namespace flowercnn {

class OptimizerSGD {
private:
    double learningRate_;

public:
    explicit OptimizerSGD(double learningRate);
    void step(Dense& layer) const;
    double learningRate() const;
};

}
