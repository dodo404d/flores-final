#pragma once

#include <string>

namespace flowercnn {

class Logger {
public:
    static void title(const std::string& message);
    static void info(const std::string& message);
    static void ok(const std::string& message);
    static void warn(const std::string& message);
    static void error(const std::string& message);
};

}
