#include <torch/torch.h>
#include <torch/types.h>

#include <iostream>

int main() {
    auto a = torch::empty({10}, torch::kInt32);
    for (int i = 0; i < 10; ++i) {
        auto b = a.data_ptr<int32_t>() + i;
        *b = i;
    }
    std::cout << a << std::endl;
}