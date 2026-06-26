#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace flowercnn {

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

    std::size_t index(std::size_t row, std::size_t col) const;

public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size() const;

    double& at(std::size_t row, std::size_t col);
    double at(std::size_t row, std::size_t col) const;

    const std::vector<double>& data() const;
    std::vector<double>& data();

    void fill(double value);
    std::vector<double> multiply(const std::vector<double>& vector) const;
    std::vector<double> transposeMultiply(const std::vector<double>& vector) const;
    std::string shape() const;
};

}
