#pragma once

#include "Matrix.h"
#include <cstdint>
#include <string_view>

/**
 * @namespace mnist
 * @brief Utilities for loading MNIST-style dataset files.
 */
namespace mnist {
    constexpr uint32_t IMAGE_MAGIC = 0x00000803;
    constexpr uint32_t LABEL_MAGIC = 0x00000801;
    constexpr int label_range = 10;

    /**
     * @brief Loads MNIST dataset images and labels into a pair of matrices.
     * @param image_path Path to the images file.
     * @param label_path Path to the labels file.
     * @param limit Maximum number of samples to load from dataset (default = 0 = all).
     * @return Pair of matrices: (images, labels).
     * @throws std::runtime_error if files cannot be loaded or read correctly.
     */
    [[nodiscard]] std::pair<Matrix, Matrix> load(
        std::string_view image_path,
        std::string_view label_path,
        int limit = 0
    );
}
