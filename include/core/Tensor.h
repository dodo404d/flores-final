#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace flowercnn {

class Tensor {
private:
    std::size_t height_;
    std::size_t width_;
    std::size_t channels_;
    std::vector<double> data_;

    std::size_t index(std::size_t y, std::size_t x, std::size_t c) const;

public:
    Tensor();
    Tensor(std::size_t height, std::size_t width, std::size_t channels, double value = 0.0);

    std::size_t height() const;
    std::size_t width() const;
    std::size_t channels() const;
    std::size_t size() const;

    double& at(std::size_t y, std::size_t x, std::size_t c);
    double at(std::size_t y, std::size_t x, std::size_t c) const;

    const std::vector<double>& data() const;
    std::vector<double>& data();

    std::vector<double> flatten() const;
    void fill(double value);

    double minValue() const;
    double maxValue() const;
    double meanValue() const;

    std::string shape() const;
};

}
