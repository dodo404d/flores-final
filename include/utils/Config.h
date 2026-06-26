#pragma once

#include <string>
#include <unordered_map>

namespace flowercnn {

class Config {
private:
    std::unordered_map<std::string, std::string> values_;

public:
    bool load(const std::string& filePath);
    std::string getString(const std::string& key, const std::string& defaultValue) const;
    int getInt(const std::string& key, int defaultValue) const;
    double getDouble(const std::string& key, double defaultValue) const;
};

}
