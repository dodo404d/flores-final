#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace flowercnn {

class LabelEncoder {
private:
    std::vector<std::string> classes_;
    std::unordered_map<std::string, int> nameToId_;

public:
    LabelEncoder();
    explicit LabelEncoder(const std::vector<std::string>& classNames);

    int encode(const std::string& className) const;
    std::string decode(int label) const;
    const std::vector<std::string>& classes() const;
    std::size_t size() const;
};

}
