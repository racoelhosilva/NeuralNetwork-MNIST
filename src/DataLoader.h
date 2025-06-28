#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "Matrix.h"
#include <string>
#include <vector>

namespace mnist {
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
