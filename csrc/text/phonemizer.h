#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "speak_lib.h"

namespace data {

// Thread local storage. The current state of ESPEAK-NG is only allowing one
// instance per-thread. Call get_thread_phonemizer to get the phonemizer's
// reference.
struct espeak_phonemizer {
    // No copy or move allowed!
    espeak_phonemizer(const espeak_phonemizer&) = delete;
    espeak_phonemizer& operator=(const espeak_phonemizer&) = delete;
    espeak_phonemizer(espeak_phonemizer&&) = delete;
    espeak_phonemizer& operator=(espeak_phonemizer&&) = delete;

    std::string phonemize_segment(std::string_view);

    // Phonemize an input sentence.
    std::string phonemize(std::string_view);

    static espeak_phonemizer& get_thread_phonemizer() {
        static thread_local auto z = espeak_phonemizer();
        return z;
    }

    ~espeak_phonemizer() { espeak_Terminate(); }

   private:
    espeak_phonemizer();
};

}  // namespace data