#include "core/Matrix.h"
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace flowercnn {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

std::size_t Matrix::index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Indice fuera de la matriz");
    }
    return row * cols_ + col;
}

std::size_t Matrix::rows() const { return rows_; }
std::size_t Matrix::cols() const { return cols_; }
std::size_t Matrix::size() const { return data_.size(); }

double& Matrix::at(std::size_t row, std::size_t col) { return data_[index(row, col)]; }
double Matrix::at(std::size_t row, std::size_t col) const { return data_[index(row, col)]; }

const std::vector<double>& Matrix::data() const { return data_; }
std::vector<double>& Matrix::data() { return data_; }

void Matrix::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::vector<double> Matrix::multiply(const std::vector<double>& vector) const {
    if (vector.size() != cols_) {
        throw std::invalid_argument("Dimension incompatible en Matrix::multiply");
    }

    std::vector<double> result(rows_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < cols_; ++j) {
            sum += at(i, j) * vector[j];
        }
        result[i] = sum;
    }
    return result;
}

std::vector<double> Matrix::transposeMultiply(const std::vector<double>& vector) const {
    if (vector.size() != rows_) {
        throw std::invalid_argument("Dimension incompatible en Matrix::transposeMultiply");
    }

    std::vector<double> result(cols_, 0.0);
    for (std::size_t j = 0; j < cols_; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < rows_; ++i) {
            sum += at(i, j) * vector[i];
        }
        result[j] = sum;
    }
    return result;
}

std::string Matrix::shape() const {
    std::ostringstream oss;
    oss << rows_ << "x" << cols_;
    return oss.str();
}

}
