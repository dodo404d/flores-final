#include "data/ImageFolderDatasetLoader.h"
#include "data/ImageProcessor.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#endif

namespace fs = std::filesystem;

namespace flowercnn {

ImageFolderDatasetLoader::ImageFolderDatasetLoader(std::size_t targetHeight, std::size_t targetWidth, int maxImagesPerClass)
    : targetHeight_(targetHeight), targetWidth_(targetWidth), maxImagesPerClass_(maxImagesPerClass) {}

bool ImageFolderDatasetLoader::isSupportedImage(const fs::path& path) const {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

bool ImageFolderDatasetLoader::loadImageWIC(const fs::path& path, Tensor& output) const {
#ifndef _WIN32
    (void)path;
    (void)output;
    return false;
#else
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool shouldUninitialize = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;

    bool ok = false;

    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory));

    if (SUCCEEDED(hr)) {
        hr = factory->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnLoad, &decoder);
    }

    if (SUCCEEDED(hr)) {
        hr = decoder->GetFrame(0, &frame);
    }

    UINT width = 0;
    UINT height = 0;
    if (SUCCEEDED(hr)) {
        hr = frame->GetSize(&width, &height);
    }

    if (SUCCEEDED(hr)) {
        hr = factory->CreateFormatConverter(&converter);
    }

    if (SUCCEEDED(hr)) {
        hr = converter->Initialize(frame, GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    }

    if (SUCCEEDED(hr) && width > 0 && height > 0) {
        const UINT channels = 3;
        const UINT stride = width * channels;
        const UINT bufferSize = stride * height;
        std::vector<BYTE> buffer(bufferSize);

        hr = converter->CopyPixels(nullptr, stride, bufferSize, buffer.data());
        if (SUCCEEDED(hr)) {
            Tensor original(height, width, channels, 0.0);
            for (UINT y = 0; y < height; ++y) {
                for (UINT x = 0; x < width; ++x) {
                    for (UINT c = 0; c < channels; ++c) {
                        original.at(y, x, c) = static_cast<double>(buffer[y * stride + x * channels + c]) / 255.0;
                    }
                }
            }
            output = ImageProcessor::resizeNearest(original, targetHeight_, targetWidth_);
            ok = true;
        }
    }

    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (shouldUninitialize) CoUninitialize();

    return ok;
#endif
}

ImageDataset ImageFolderDatasetLoader::load(const std::string& rootFolder, const LabelEncoder& encoder) const {
    ImageDataset dataset;
    fs::path root(rootFolder);

    if (!fs::exists(root)) {
        Logger::warn("No existe la carpeta del dataset: " + rootFolder);
        return dataset;
    }

    for (const auto& className : encoder.classes()) {
        fs::path classFolder = root / className;
        int loadedForClass = 0;

        if (!fs::exists(classFolder) || !fs::is_directory(classFolder)) {
            Logger::warn("No existe la carpeta de clase: " + classFolder.string());
            continue;
        }

        for (const auto& entry : fs::directory_iterator(classFolder)) {
            if (!entry.is_regular_file()) continue;
            if (!isSupportedImage(entry.path())) continue;
            if (maxImagesPerClass_ > 0 && loadedForClass >= maxImagesPerClass_) break;

            Tensor image;
            if (loadImageWIC(entry.path(), image)) {
                ImageSample sample;
                sample.image = image;
                sample.label = encoder.encode(className);
                sample.className = className;
                sample.filePath = entry.path().string();
                dataset.add(sample);
                loadedForClass++;
            }
        }

        std::ostringstream oss;
        oss << "Clase " << className << ": " << loadedForClass << " imagen(es) cargada(s).";
        Logger::info(oss.str());
    }

    return dataset;
}

}
