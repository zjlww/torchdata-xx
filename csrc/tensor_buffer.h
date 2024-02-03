#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>

#include "types.h"

namespace data {
struct TensorBuffer {
    int64_t dim{0};
    IValue buffer{};

    TensorBuffer(){};
    TensorBuffer(int64_t dim) : dim{dim} {};

    void push(Tensor t) {
        if (buffer.isNone()) {
            buffer = t;
        } else {
            buffer = torch::cat({buffer.toTensor(), t}, dim);
        }
    }

    int size() {
        if (buffer.isNone()) {
            return 0;
        } else {
            return buffer.toTensor().size(dim);
        }
    }

    Tensor pop(int64_t n) {
        auto b = buffer.toTensor();
        if (n > b.size(dim)) {
            throw std::out_of_range("TensorBuffre pop out of range");
        }
        auto a = b.slice(dim, 0, n);
        auto c = b.slice(dim, n);
        buffer = c;
        return a;
    }
};
}  // namespace data