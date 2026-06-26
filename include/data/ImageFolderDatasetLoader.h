#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "data/Dataset.h"
#include "data/LabelEncoder.h"

namespace flowercnn {

class ImageFolderDatasetLoader {
private:
    std::size_t targetHeight_;
    std::size_t targetWidth_;
    int maxImagesPerClass_;

    bool isSupportedImage(const std::filesystem::path& path) const;
    bool loadImageWIC(const std::filesystem::path& path, Tensor& output) const;

public:
    ImageFolderDatasetLoader(std::size_t targetHeight, std::size_t targetWidth, int maxImagesPerClass);

    ImageDataset load(const std::string& rootFolder, const LabelEncoder& encoder) const;
};

}
