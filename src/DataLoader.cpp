#include "DataLoader.h"
#include <fstream>

static int32_t read_header_int(std::ifstream& file) {
    uint8_t value[4];
    file.read(reinterpret_cast<char*>(value), 4);
    return value[0] << 24
        | value[1] << 16
        | value[2] << 8
        | value[3];
}

std::pair<Matrix, Matrix> mnist::load(
    const std::string& image_path,
    const std::string& label_path,
    int limit
) {
    std::ifstream images { image_path, std::ios::binary };
    if (!images) { throw std::runtime_error("could not open MNIST images file"); }

    std::ifstream labels { label_path, std::ios::binary };
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
    
    std::vector<uint8_t> buffer(rows * cols);
    uint8_t label;

    for (int col { 0 }; col < limit; ++col) {
        images.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        labels.read(reinterpret_cast<char*>(&label), 1);

        for (int row { 0 }; row < X.rows(); ++row) {
            X[row, col] = buffer[row] / 255.0;
        }

        y[label, col] = 1.0;
    }

    return {X, y};
}
