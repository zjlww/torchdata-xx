#pragma once
#include <sox.h>
#include <soxr.h>

#include <mutex>
#include <string>
#include <string_view>
#include <variant>

#include "types.h"

namespace data {

struct AudioFile {
    sox_format_t* pt{};
    sox_rate_t rate{};
    size_t length{};
    unsigned channels{};
    std::string path{};
    AudioFile() = default;
    explicit AudioFile(std::string_view path);

    [[nodiscard]] Tensor wave() const;
    ~AudioFile() noexcept;
};

struct AudioMemoryFile : AudioFile {
    explicit AudioMemoryFile(void* data, size_t size);
    ~AudioMemoryFile() noexcept = default;
};

// Read an audio from a file.
// Returns the int32 encoded waveform and the sampling rate.
std::pair<Tensor, double> readAudio(std::string_view path);

// Read an audio from a memory buffer.
// Returns the int32 encoded waveform and the sampling rate.
std::pair<Tensor, double> readAudioMemory(void* data, size_t size);

// Resample an audio from in_rate to out_rate.
// Returns the resampled audio.
// Implemented with soxr, you can change the parameters in the source code.
Tensor resample(Tensor inWave, double inRate, double outRate);

// Save a waveform to target path. wave is IntTensor[nSample, nChannel].
void wavSavePCM32Mono(Tensor wave, std::string_view path, sox_rate_t rate);

}  // namespace data