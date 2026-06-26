#include "utils/Logger.h"
#include <iostream>

namespace flowercnn {

void Logger::title(const std::string& message) {
    std::cout << "\n============================================================\n";
    std::cout << message << "\n";
    std::cout << "============================================================\n";
}

void Logger::info(const std::string& message) {
    std::cout << "[INFO] " << message << "\n";
}

void Logger::ok(const std::string& message) {
    std::cout << "[OK] " << message << "\n";
}

void Logger::warn(const std::string& message) {
    std::cout << "[WARN] " << message << "\n";
}

void Logger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << "\n";
}

}
