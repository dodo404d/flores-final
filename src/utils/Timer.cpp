#include "utils/Timer.h"

namespace flowercnn {

void Timer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsedMilliseconds() const {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start_;
    return elapsed.count();
}

}
