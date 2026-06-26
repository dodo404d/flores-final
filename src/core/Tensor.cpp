#include "core/Tensor.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace flowercnn {

Tensor::Tensor() : height_(0), width_(0), channels_(0) {}

Tensor::Tensor(std::size_t height, std::size_t width, std::size_t channels, double value)
    : height_(height), width_(width), channels_(channels), data_(height * width * channels, value) {}

std::size_t Tensor::index(std::size_t y, std::size_t x, std::size_t c) const {
    if (y >= height_ || x >= width_ || c >= channels_) {
        throw std::out_of_range("Indice fuera del tensor");
    }
    return (y * width_ + x) * channels_ + c;
}

std::size_t Tensor::height() const { return height_; }
std::size_t Tensor::width() const { return width_; }
std::size_t Tensor::channels() const { return channels_; }
std::size_t Tensor::size() const { return data_.size(); }

double& Tensor::at(std::size_t y, std::size_t x, std::size_t c) { return data_[index(y, x, c)]; }
double Tensor::at(std::size_t y, std::size_t x, std::size_t c) const { return data_[index(y, x, c)]; }

const std::vector<double>& Tensor::data() const { return data_; }
std::vector<double>& Tensor::data() { return data_; }

std::vector<double> Tensor::flatten() const { return data_; }

void Tensor::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

double Tensor::minValue() const {
    if (data_.empty()) return 0.0;
    return *std::min_element(data_.begin(), data_.end());
}

double Tensor::maxValue() const {
    if (data_.empty()) return 0.0;
    return *std::max_element(data_.begin(), data_.end());
}

double Tensor::meanValue() const {
    if (data_.empty()) return 0.0;
    double sum = std::accumulate(data_.begin(), data_.end(), 0.0);
    return sum / static_cast<double>(data_.size());
}

std::string Tensor::shape() const {
    std::ostringstream oss;
    oss << height_ << "x" << width_ << "x" << channels_;
    return oss.str();
}

}
