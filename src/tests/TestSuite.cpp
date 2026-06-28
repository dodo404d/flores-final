#include "tests/TestSuite.h"
#include "utils/Logger.h"
#include "core/Tensor.h"
#include "core/Matrix.h"
#include "layers/Dense.h"
#include "layers/ReLU.h"
#include "layers/SoftmaxCrossEntropy.h"
#include "core/RandomInitializer.h"

#include <cmath>
#include <vector>
#include <sstream>

namespace flowercnn {

static bool almostEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

bool TestSuite::runAll() {
    Logger::title("Pruebas internas del Bloque 1");
    bool ok = true;

    try {
        Tensor t(2, 3, 1, 0.5);
        ok = ok && (t.height() == 2 && t.width() == 3 && t.channels() == 1 && almostEqual(t.at(1, 2, 0), 0.5));
        Logger::ok("Tensor: dimensiones y acceso correctos.");
    } catch (...) {
        Logger::error("Fallo en prueba Tensor.");
        ok = false;
    }

    try {
        Matrix m(2, 2, 0.0);
        m.at(0, 0) = 1.0; m.at(0, 1) = 2.0;
        m.at(1, 0) = 3.0; m.at(1, 1) = 4.0;
        auto out = m.multiply({1.0, 1.0});
        ok = ok && almostEqual(out[0], 3.0) && almostEqual(out[1], 7.0);
        Logger::ok("Matrix: multiplicacion correcta.");
    } catch (...) {
        Logger::error("Fallo en prueba Matrix.");
        ok = false;
    }

    try {
        RandomInitializer::setSeed(42);
        Dense dense(3, 5);
        auto out = dense.forward({0.1, 0.2, 0.3});
        ok = ok && (out.size() == 5);
        Logger::ok("Dense: salida con dimension correcta.");
    } catch (...) {
        Logger::error("Fallo en prueba Dense.");
        ok = false;
    }

    try {
        ReLU relu;
        std::vector<double> reluInput = {-1.0, 2.0, -0.5, 3.0};
        auto out = relu.forward(reluInput);
        
        ok = ok && almostEqual(out[0], 0.0) && almostEqual(out[1], 2.0) && almostEqual(out[2], 0.0) && almostEqual(out[3], 3.0);
        Logger::ok("ReLU: activacion correcta.");
    } catch (...) {
        Logger::error("Fallo en prueba ReLU.");
        ok = false;
    }

    try {
        SoftmaxCrossEntropy loss;
        double value = loss.forward({1.0, 2.0, 3.0}, 2);
        double sum = 0.0;
        for (double p : loss.probabilities()) sum += p;
        ok = ok && almostEqual(sum, 1.0, 1e-8) && value > 0.0;
        Logger::ok("SoftmaxCrossEntropy: probabilidades validas.");
    } catch (...) {
        Logger::error("Fallo en prueba SoftmaxCrossEntropy.");
        ok = false;
    }

    if (ok) Logger::ok("Todas las pruebas internas pasaron correctamente.");
    else Logger::warn("Algunas pruebas internas fallaron.");

    return ok;
}

}
