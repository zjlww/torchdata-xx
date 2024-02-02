#include <iostream>

#include "dataset.h"
#include "types.h"

int main() {
    auto dt = data::loadShard("./shit.pt");
    auto item = (*dt)["123"];
    std::cout << "successful" << std::endl;
    std::cout << std::get<data::Tensor>(item["x"]) << std::endl;
    return 0;
}