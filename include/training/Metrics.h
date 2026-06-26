#pragma once

#include <vector>

namespace flowercnn {

class Metrics {
public:
    static int argmax(const std::vector<double>& values);
    static double accuracy(const std::vector<int>& predictions, const std::vector<int>& labels);
};

}
