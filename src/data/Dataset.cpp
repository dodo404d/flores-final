#include "data/Dataset.h"

namespace flowercnn {

void ImageDataset::add(const ImageSample& sample) {
    samples_.push_back(sample);
}

std::size_t ImageDataset::size() const {
    return samples_.size();
}

bool ImageDataset::empty() const {
    return samples_.empty();
}

const std::vector<ImageSample>& ImageDataset::samples() const {
    return samples_;
}

std::map<std::string, int> ImageDataset::countByClass() const {
    std::map<std::string, int> counts;
    for (const auto& sample : samples_) {
        counts[sample.className]++;
    }
    return counts;
}

}
