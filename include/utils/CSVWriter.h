#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace flowercnn {

class CSVWriter {
private:
    std::ofstream file_;

public:
    explicit CSVWriter(const std::string& filePath);
    bool isOpen() const;
    void writeRow(const std::vector<std::string>& columns);
};

}
