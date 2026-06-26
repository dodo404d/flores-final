#pragma once

#include <chrono>

namespace flowercnn {

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;

public:
    void start();
    double elapsedMilliseconds() const;
};

}
