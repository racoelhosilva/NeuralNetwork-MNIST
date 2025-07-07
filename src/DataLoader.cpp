#include "DataLoader.h"
#include <array>
#include <format>
#include <fstream>
#include <stdexcept>

/**
 * @brief Reads a 4-byte big-endian integer from the file.
 */
[[nodiscard]] static int32_t read_header_int(std::ifstream& file) {
    std::array<uint8_t, 4> value;
    file.read(reinterpret_cast<char*>(value.data()), 4);
    return value[0] << 24
        | value[1] << 16
        | value[2] << 8
        | value[3];
}

std::pair<Matrix, Matrix> mnist::load(
    std::string_view image_path,
    std::string_view label_path,
    int limit
) {
    std::ifstream images { std::string(image_path), std::ios::binary };
    if (!images) { throw std::runtime_error("could not open MNIST images file"); }

    std::ifstream labels { std::string(label_path), std::ios::binary };
    if (!labels) { throw std::runtime_error("could not open MNIST labels file"); }

    if (read_header_int(images) != mnist::IMAGE_MAGIC || 
        read_header_int(labels) != mnist::LABEL_MAGIC) {
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

    Matrix X(rows * cols, limit);
    Matrix y(mnist::label_range, limit);
    
    std::vector<uint8_t> buffer(static_cast<size_t>(rows * cols));
    uint8_t label;

    for (int col { 0 }; col < limit; ++col) {
        if (!images.read(reinterpret_cast<char*>(buffer.data()), 
            static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error(
                    std::format("failed to read image data at sample {}", col));
        }
        if (!labels.read(reinterpret_cast<char*>(&label), 1)) {
            throw std::runtime_error(
                std::format("failed to read label data at sample {}", col));
        }

        for (int row { 0 }; row < X.rows(); ++row) {
            X[row, col] = buffer[row] / 255.0;
        }

        y[label, col] = 1.0;
    }

    return {X, y};
}
