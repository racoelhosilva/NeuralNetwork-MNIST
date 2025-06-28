#include "DataLoader.h"
#include <cstdint>
#include <fstream>

static int32_t read_header_int(std::ifstream& file) {
    uint8_t value[4];
    file.read(reinterpret_cast<char*>(value), 4);
    return value[0] << 24
        | value[1] << 16
        | value[2] << 8
        | value[3];
}

namespace mnist {
    std::vector<Record> load(
        const std::string& image_path,
        const std::string& label_path,
        int limit
    ) {
        std::ifstream images { image_path, std::ios::binary };
        if (!images) { throw std::runtime_error("could not open MNIST images file"); }

        std::ifstream labels { label_path, std::ios::binary };
        if (!labels) { throw std::runtime_error("could not open MNIST labels file"); }

        if (read_header_int(images) != 0x00000803 || 
            read_header_int(labels) != 0x00000801) {
            throw std::runtime_error("unmatched MNIST header magic numbers");
        }

        int32_t image_count = read_header_int(images);
        int32_t label_count = read_header_int(labels);
        if (image_count != label_count) {
            throw std::runtime_error("unmatched number of images and labels");
        }

        int rows = static_cast<int>(read_header_int(images));
        int cols = static_cast<int>(read_header_int(images));
        if (limit <= 0 || limit > static_cast<int>(image_count)) {
            limit = static_cast<int>(image_count);
        }

        std::vector<uint8_t> buffer(rows * cols);
        uint8_t label;
        std::vector<mnist::Record> records;
        records.reserve(limit);

        while (limit--) {
            images.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            labels.read(reinterpret_cast<char*>(&label), 1);

            std::vector<double> image(rows * cols);
            for (size_t i = 0; i < image.size(); ++i) {
                image[i] = buffer[i] / 255.0;
            }

            Matrix matrix(rows, cols, std::move(image));

            records.push_back({std::move(matrix), static_cast<int>(label)});
        }

        return records;
    }
}
