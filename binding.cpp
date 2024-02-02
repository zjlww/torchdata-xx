#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

#include <memory>
#include <string_view>

#include "dataset.h"
#include "types.h"

namespace py = pybind11;
using namespace pybind11::literals;

inline void bindDataset(pybind11::module& m) {
    auto Dataset =
        py::class_<data::Dataset, data::DatasetHandle>(m, "Dataset")
            .def("__len__", &data::Dataset::size)
            .def("__contains__", &data::Dataset::contains, py::arg("key"))
            .def("__getitem__", &data::Dataset::operator[], py::arg("key"))
            .def_readonly("keys", &data::Dataset::keys)
            .def("zip", &data::Dataset::zip, py::arg("other"))
            .def("merge", &data::Dataset::merge, py::arg("other"))
            .def("prefix", &data::Dataset::prefix, py::arg("prefix"))
            .def("filter", &data::Dataset::filter, py::arg("pred"))
            .def("map", &data::Dataset::map, py::arg("func"))
            // .def("sample", &data::Dataset::sample)
            .def("to_map", &data::Dataset::toMap);
    m.def("loadShard", data::loadShard, py::arg("path"));
}

PYBIND11_MODULE(torchdataxx, m) {
    m.doc() = "TorchData-XX Python Binding Module";
    bindDataset(m);
}
