#include "utils/CSVWriter.h"

namespace flowercnn {

CSVWriter::CSVWriter(const std::string& filePath) : file_(filePath) {}

bool CSVWriter::isOpen() const {
    return file_.is_open();
}

void CSVWriter::writeRow(const std::vector<std::string>& columns) {
    if (!file_.is_open()) return;

    for (std::size_t i = 0; i < columns.size(); ++i) {
        file_ << columns[i];
        if (i + 1 < columns.size()) {
            file_ << ",";
        }
    }
    file_ << "\n";
}

}
