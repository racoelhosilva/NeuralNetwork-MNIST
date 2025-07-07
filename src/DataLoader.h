#pragma once

#include "Matrix.h"
#include <cstdint>
#include <string>
#include <utility>

namespace mnist {
    constexpr uint32_t IMAGE_MAGIC = 0x00000803;
    constexpr uint32_t LABEL_MAGIC = 0x00000801;
    const int label_range = 10;

    std::pair<Matrix, Matrix> load(
        const std::string& image_path,
        const std::string& label_path,
        int limit = 0
    );
}
