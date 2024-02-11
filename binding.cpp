#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

#include <memory>
#include <string_view>

#include "audio.h"
#include "dataset.h"
#include "functional.h"
#include "tensor_utils.h"
#include "types.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace data {

// Binding for csrc/functional.h
inline void bindFunctional(py::module& m) {
    auto F = m.def_submodule("functional", "Functionals");
    auto mItemTransform =
        py::class_<ItemTransform, ItemTransformHandle>(m, "ItemTransform")
            .def("__call__", &ItemTransform::operator(), py::arg("item"));
    auto mItemPredicate =
        py::class_<ItemPredicate, ItemPredicateHandle>(m, "ItemPredicate")
            .def("__call__", &ItemPredicate::operator(), py::arg("item"));
    auto mKeyPredicate =
        py::class_<KeyPredicate, KeyPredicateHandle>(m, "KeyPredicate")
            .def("__call__", &KeyPredicate::operator(), py::arg("key"));
    F.def("roll", data::roll, py::arg("key"), py::arg("dim"), py::arg("shift"));
    F.def("randomRoll", randomRoll, py::arg("key"), py::arg("dim"),
          py::arg("shiftMin"), py::arg("shiftMax"));
    F.def("rightPadSequenceFrame", rightPadSequenceFrame, py::arg("key"),
          py::arg("frameKey"), py::arg("dim"), py::arg("frameSize"));
    F.def("rightTruncateSequenceFrame", rightTruncateSequenceFrame,
          py::arg("key"), py::arg("frameKey"), py::arg("dim"),
          py::arg("frameSize"));
    F.def("addInt64", addInt64, py::arg("keyA"), py::arg("keyB"),
          py::arg("keyC"), py::arg("bias"));
    F.def("readFile", readFile, py::arg("pathKey"), py::arg("textKey"));
    F.def("readAudioTransform", readAudioTransform, py::arg{"pathKey"},
          py::arg("waveKey"), py::arg("srKey"), py::arg("asFloat32"));
}

// Binding for csrc/dataset.h
inline void bindDataset(py::module& m) {
    auto mDataset =
        py::class_<Dataset, DatasetHandle>(m, "Dataset")
            .def("__len__", &Dataset::size)
            .def("__contains__", &Dataset::contains, py::arg("key"))
            .def("__getitem__", &Dataset::operator[], py::arg("key"))
            .def("getItem", &Dataset::getItem, py::arg("idx"))
            .def_readonly("keys", &Dataset::keys)
            .def("map", &Dataset::map, py::arg("func"))
            .def("filter", &Dataset::filter, py::arg("pred"))
            .def("zip", &Dataset::zip, py::arg("other"))
            .def("merge", &Dataset::merge, py::arg("other"))
            .def("prefix", &Dataset::prefix, py::arg("prefix"))
            .def("sample", &Dataset::sample)
            .def("permuteSample", &Dataset::permuteSample)
            .def("toMap", &Dataset::toMap);
    m.def("loadShard", loadShard, py::arg("path"));
    m.def("immediateDataset", immediateDataset, py::arg("items"));
}

// Binding for csrc/sampler.h
inline void bindSampler(py::module& m) {
    auto mSampler =
        py::class_<Sampler, SamplerHandle>(m, "Sampler")
            .def("sample", &Sampler::sample)
            .def("map", &Sampler::map, py::arg("func"))
            .def("filter", &Sampler::filter, py::arg("pred"))
            .def("queue", &Sampler::queue, py::arg("nThreads"),
                 py::arg("queueSize"))
            .def("batch", &Sampler::batch, py::arg("batchSize"))
            .def("zipDataset", &Sampler::zipDataset, py::arg("dataset"),
                 py::arg("keyKey"))
            .def("segment", &Sampler::segment, py::arg("bufferKey"),
                 py::arg("segmentSize"), py::arg("dim"))
            .def("segmentSlicing", &Sampler::segmentSlicing,
                 py::arg("bufferKey"), py::arg("segmentSize"), py::arg("dim"))
            .def("segmentClasswise", &Sampler::segmentClasswise,
                 py::arg("bufferKey"), py::arg("classKey"),
                 py::arg("segmentSize"), py::arg("dim"))
            .def("bucket", &Sampler::bucket, py::arg("sortKey"),
                 py::arg("partition"))
            .def("sampleShard", &Sampler::sampleShard, py::arg("shardPathKey"),
                 py::arg("shardIDKey"), py::arg("samplesPerShard"));

    auto mBatchSampler =
        py::class_<BatchSampler, BatchSamplerHandle>(m, "BatchSampler")
            .def("sample", &BatchSampler::sample)
            .def("stack", &BatchSampler::stack)
            .def("flatten", &BatchSampler::flatten);
}

// Binding for csrc/audio.h
inline void bindAudio(py::module& m) {
    auto A = m.def_submodule("audio", "Audio Utilities.");
    A.def("readAudio", readAudio, py::arg("path"));
    A.def("resample", resample, py::arg("inWave"), py::arg("inRate"),
          py::arg("outRate"));
    A.def("wavSavePCM", wavSavePCM, py::arg("wave"), py::arg("path"),
          py::arg("sr"), py::arg("bits"));
}

// Binding for csrc//tensor_buffer.h
inline void bindTensorBuffer(py::module& m) {
    auto mBuffer = py::class_<TensorBuffer>(m, "TensorBuffer")
                       .def(py::init<int64_t>())
                       .def("push", &TensorBuffer::push, py::arg("t"))
                       .def("size", &TensorBuffer::size)
                       .def("pop", &TensorBuffer::pop, py::arg("n"));
}

}  // namespace data

PYBIND11_MODULE(torchdataxx_C, m) {
    m.doc() = "TorchData-XX Python Binding Module";
    data::bindFunctional(m);
    data::bindDataset(m);
    data::bindSampler(m);
    data::bindAudio(m);
    data::bindTensorBuffer(m);
}
