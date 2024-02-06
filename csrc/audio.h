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
// waveforms in shape [nSample, nChannel].
// Returns the int32 encoded waveform and the sampling rate.
std::pair<Tensor, double> readAudio(std::string_view path);

// Read an audio from a memory buffer.
// waveforms in shape [nSample, nChannel].
// Returns the int32 encoded waveform and the sampling rate.
std::pair<Tensor, double> readAudioMemory(void* data, size_t size);

// Resample an audio from in_rate to out_rate.
// waveforms in shape [nSample, nChannel].
// Returns the resampled audio.
// Implemented with soxr, you can change the parameters in the source code.
Tensor resample(Tensor inWave, double inRate, double outRate);

// Save a waveform to target path. wave is IntTensor[nSample, nChannel].
void wavSavePCM(Tensor wave, std::string_view path, sox_rate_t sr,
                unsigned int bits);

ItemTransformHandle readAudioTransform(std::string path_key,
                                       std::string wave_key, std::string sr_key,
                                       bool asFloat32);
}  // namespace data