#include <iostream>

#include "audio.h"

int main() {
    auto [w, sr] = data::readAudio(
        "/mass/1/LibriTTS-R/dev-other/1585/157660/"
        "1585_157660_000007_000000.wav");
    std::cout << w << std::endl;
}