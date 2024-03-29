#include "audio.h"

#include <sox.h>
#include <soxr.h>
#include <torch/types.h>

#include <climits>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "types.h"

namespace data {

std::once_flag once_format_init{};

AudioFile::AudioFile(std::string_view path) : path{path} {
    std::call_once(once_format_init, [] { sox_format_init(); });
    pt = sox_open_read(path.data(), nullptr, nullptr, nullptr);
    if (pt == nullptr) {
        throw std::runtime_error("failed to open file at path: " + this->path);
    }
    length = pt->signal.length;
    rate = pt->signal.rate;
    channels = pt->signal.channels;
}

AudioMemoryFile::AudioMemoryFile(void* data, size_t size) {
    std::call_once(once_format_init, [] { sox_format_init(); });
    pt = sox_open_mem_read(data, size, nullptr, nullptr, nullptr);
    if (pt == nullptr) {
        throw std::runtime_error("failed to open memory file");
    }
    length = pt->signal.length;
    rate = pt->signal.rate;
    channels = pt->signal.channels;
}

// For multi-channels audio, sox_read will return the channels interleaved.
// wave (IntTensor): [length, channels].
Tensor AudioFile::wave() const {
    Tensor wave = torch::empty(
        {static_cast<int>(length), static_cast<int>(channels)}, torch::kInt32);
    auto cnt = sox_read(pt, wave.data_ptr<int32_t>(), length * channels);

    if (cnt == 0) {
        throw std::runtime_error("failed to read file at path: " + this->path);
    }

    return wave;
}

AudioFile::~AudioFile() noexcept { auto result = sox_close(pt); }

// There are a lot of specifications here, you may want to tune them.
// Typically precision = 16, the higher the better.
Tensor resample(Tensor inWave, double inRate, double outRate) {
    unsigned channels = inWave.size(1);
    size_t in_length = inWave.size(0);
    size_t out_length = static_cast<size_t>(in_length * outRate / inRate + .5);
    Tensor out_wave =
        torch::empty({static_cast<int>(out_length), channels}, torch::kInt32);

    soxr_io_spec_t const io_spec{
        .itype = SOXR_INT32_I, .otype = SOXR_INT32_I, .scale = 1.0};

    soxr_quality_spec_t const quality_spec{.precision = 20,
                                           .phase_response = 50,
                                           .passband_end = 0.95,
                                           .stopband_begin = 1.0};

    soxr_oneshot(inRate, outRate, channels, inWave.data_ptr<int32_t>(),
                 in_length, nullptr, out_wave.data_ptr<int32_t>(), out_length,
                 nullptr, &io_spec, &quality_spec, nullptr);
    return out_wave;
}

std::pair<Tensor, double> readAudio(std::string_view path) {
    auto file = AudioFile(path);
    auto wave = file.wave();
    return {wave, file.rate};
}

std::pair<Tensor, double> readAudioMemory(void* data, size_t size) {
    auto file = AudioMemoryFile(data, size);
    auto wave = file.wave();
    return {wave, file.rate};
}

// wave (IntTensor): [nSample, nChannel].
void wavSavePCM(Tensor wave, std::string_view path, sox_rate_t sr,
                unsigned int bits) {
    unsigned channels = wave.size(1);
    size_t length = wave.size(0);
    sox_signalinfo_t signalinfo{
        .rate = sr, .channels = channels, .precision = bits, .length = length};

    auto pt = sox_open_write(path.data(), &signalinfo, nullptr, "wav", nullptr,
                             [](const char* path) { return sox_true; });

    if (pt == nullptr) {
        throw std::runtime_error("failed to open file at path: " +
                                 std::string(path));
    }

    size_t cnt = sox_write(pt, wave.data_ptr<int32_t>(), length);
    if (cnt == 0) {
        throw std::runtime_error("failed to write file at path: " +
                                 std::string(path));
    }
}

struct ReadAudioTransform final : public ItemTransform {
    std::string pathKey;
    std::string waveKey;
    std::string srKey;
    bool asFloat32;

    ReadAudioTransform(std::string pathKey, std::string waveKey,
                       std::string srKey, bool asFloat32)
        : pathKey{pathKey},
          waveKey{waveKey},
          srKey{srKey},
          asFloat32{asFloat32} {}
    Item operator()(Item item) override {
        auto [w, sr] = readAudio(std::get<std::string>(item[pathKey]));
        if (asFloat32) {
            w = w.to(torch::kFloat64) / double(INT_MAX);
            w = w.to(torch::kFloat32);
        }
        item[waveKey] = w;
        item[srKey] = sr;
        return item;
    }
};

ItemTransformHandle readAudioTransform(std::string pathKey, std::string waveKey,
                                       std::string srKey, bool asFloat32) {
    return std::make_shared<ReadAudioTransform>(pathKey, waveKey, srKey,
                                                asFloat32);
}

struct AudioPCM16AsFloat32Transform final : public ItemTransform {
    std::string waveKey;
    bool asFloat32;

    AudioPCM16AsFloat32Transform(std::string waveKey) : waveKey{waveKey} {}
    Item operator()(Item item) override {
        auto w = std::get<Tensor>(item[waveKey]);
        if (asFloat32) {
            w = w.to(torch::kFloat64) / 32768.0;
            w = w.to(torch::kFloat32);
        }
        item[waveKey] = w;
        return item;
    }
};

ItemTransformHandle audioPCM16AsFloat32(std::string waveKey) {
    return std::make_shared<AudioPCM16AsFloat32Transform>(waveKey);
}

}  // namespace data