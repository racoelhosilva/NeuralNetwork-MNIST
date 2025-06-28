#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "Matrix.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mnist {
    constexpr uint32_t IMAGE_MAGIC = 0x00000803;
    constexpr uint32_t LABEL_MAGIC = 0x00000801;

    struct Record {
        Matrix input;
        int label;
    };

    std::vector<Record> load(
        const std::string& image_path,
        const std::string& label_path,
        int limit = 0
    );
}

#endif
