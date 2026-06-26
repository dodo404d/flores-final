#pragma once

#include <vector>
#include <string>
#include <map>
#include "core/Tensor.h"

namespace flowercnn {

struct ImageSample {
    Tensor image;
    int label;
    std::string className;
    std::string filePath;
};

class ImageDataset {
private:
    std::vector<ImageSample> samples_;

public:
    void add(const ImageSample& sample);
    std::size_t size() const;
    bool empty() const;

    const std::vector<ImageSample>& samples() const;
    std::map<std::string, int> countByClass() const;
};

}
