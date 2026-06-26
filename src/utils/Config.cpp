#include "utils/Config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace flowercnn {

static std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) start++;

    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) end--;

    return value.substr(start, end - start);
}

bool Config::load(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        values_[key] = value;
    }

    return true;
}

std::string Config::getString(const std::string& key, const std::string& defaultValue) const {
    auto it = values_.find(key);
    return it != values_.end() ? it->second : defaultValue;
}

int Config::getInt(const std::string& key, int defaultValue) const {
    auto it = values_.find(key);
    if (it == values_.end()) return defaultValue;
    try { return std::stoi(it->second); }
    catch (...) { return defaultValue; }
}

double Config::getDouble(const std::string& key, double defaultValue) const {
    auto it = values_.find(key);
    if (it == values_.end()) return defaultValue;
    try { return std::stod(it->second); }
    catch (...) { return defaultValue; }
}

}
