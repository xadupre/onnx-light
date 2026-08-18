#include "_onnxpy_node_list.h"
#include "_onnxpyprotoop.h"
#include "onnx.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_crypt.h"
#include "onnx_helper.h"
#include "onnx_lib/onnx-data.pb.h"
#include "onnx_verify.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <type_traits>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {
constexpr size_t MAX_SHORT_REPR_LENGTH = 60;

// Adapts a Python callable to ParseOptions::raw_data_callback. The callable is invoked as
// ``fn(tensor, graph)`` for every parsed TensorProto that has raw_data and must return either
// ``None`` (ownership unchanged) or a zero-argument callable used as the tensor's raw_data
// deleter. ``graph`` is the parent GraphProto or ``None`` when the tensor is parsed on its own.
// The Python object is held by value so the deleter retrieved by std::function::target keeps a
// live reference, and the GIL is reacquired around every Python call.
struct PyRawDataCallback {
  nb::object fn;
  std::function<void()> operator()(TensorProto &tensor, GraphProto *graph) const {
    nb::gil_scoped_acquire gil;
    nb::object py_graph = graph == nullptr ? nb::none() : nb::cast(graph, nb::rv_policy::reference);
    nb::object result = fn(nb::cast(&tensor, nb::rv_policy::reference), py_graph);
    if (result.is_none()) {
      return {};
    }
    nb::object deleter = result;
    return [deleter]() {
      nb::gil_scoped_acquire gil;
      deleter();
    };
  }
};

// Adapts a Python callable to SerializeOptions::raw_data_callback. The callable is invoked as
// ``fn(tensor, graph, buffer, size_only)``:
// - when ``size_only`` is True, ``buffer`` is None and the callback must return the number of
//   bytes it will write for that tensor;
// - when ``size_only`` is False, ``buffer`` is a writable 1-D uint8 NumPy view owned by
//   onnx-light; the callback may update tensor metadata in place, must write the serialized bytes
//   into that buffer, and must return the same size again.
// ``graph`` is the tensor's parent GraphProto (from a working copy of the model) or ``None``.
// The Python object is held by value so the std::function target keeps the callable alive, and
// the GIL is reacquired around every call.
struct PySerializeRawDataCallback {
  nb::object fn;
  int64_t operator()(TensorProto &tensor, GraphProto *graph, uint8_t *buffer, size_t buffer_size,
                     bool size_only) const {
    nb::gil_scoped_acquire gil;
    nb::object py_buffer;
    if (size_only || buffer == nullptr) {
      py_buffer = nb::none();
    } else {
      // The writable view borrows C++-managed storage for the duration of the callback only,
      // so the capsule deliberately uses a no-op deleter.
      nb::capsule owner(static_cast<void *>(buffer), [](void *) noexcept {});
      py_buffer =
          nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(buffer, {buffer_size}, owner));
    }
    nb::object py_graph = graph == nullptr ? nb::none() : nb::cast(graph, nb::rv_policy::reference);
    return nb::cast<int64_t>(
        fn(nb::cast(&tensor, nb::rv_policy::reference), py_graph, py_buffer, size_only));
  }
};

// Adapts a Python callable to ParseOptions/SerializeOptions::node_callback. The callable is
// invoked as ``fn(node, graph)`` for every NodeProto once it has been parsed (or right before it
// is serialized) and receives the NodeProto and its parent GraphProto by reference so it may
// inspect or edit the node in place and locate its surrounding graph. The Python object is held
// by value so the std::function target keeps the callable alive, and the GIL is reacquired around
// every call.
struct PyNodeCallback {
  nb::object fn;
  void operator()(NodeProto &node, GraphProto &graph) const {
    nb::gil_scoped_acquire gil;
    fn(nb::cast(&node, nb::rv_policy::reference), nb::cast(&graph, nb::rv_policy::reference));
  }
};

// Reusable ``raw_data_callback`` that keeps the default C++ allocation (tensor ownership
// unchanged) while letting users observe parsing progress. When called it optionally forwards
// the freshly parsed TensorProto to a user callable invoked as ``on_tensor(tensor)`` (for
// example to print progress) and always returns ``None``, so the tensor's ``raw_data`` is left
// to the default allocator. Subclasses may override ``__call__`` for richer behavior.
struct RawDataCallback {
  nb::object on_tensor;
};

ModelProto MakeOwnedModelProtoCopy(const ModelProto &model) {
  // Fully reparse through bytes to ensure every borrowed span in the model
  // becomes owned before serialization paths that may mutate buffers/metadata.
  std::string serialized;
  model.SerializeToString(serialized);
  ModelProto owned;
  owned.ParseFromString(serialized);
  return owned;
}

// --------------------------------------------------------------------------
// Convenience builder helpers exposed as proto methods. The Python-level
// type inference for attributes/tensors lives in ``onnx_light.onnx_proto._helper``
// and ``._numpy_helper``; these wrappers import those modules on demand so we
// don't duplicate the (rich) inference logic in C++ while still exposing the
// helpers as native methods on the bound proto classes.
// --------------------------------------------------------------------------

inline nb::module_ ImportHelper() { return nb::module_::import_("onnx_light.onnx_proto._helper"); }

inline nb::module_ ImportNumpyHelper() {
  return nb::module_::import_("onnx_light.onnx_proto._numpy_helper");
}

// Build a ValueInfoProto from either an existing ValueInfoProto, a bare name,
// or a (name, elem_type, shape) triple via helper.make_tensor_value_info.
ValueInfoProto MakeValueInfoForPy(nb::object name_or_proto, nb::object elem_type, nb::object shape,
                                  nb::object doc_string) {
  if (nb::isinstance<ValueInfoProto>(name_or_proto)) {
    if (!elem_type.is_none() || !shape.is_none()) {
      throw nb::value_error("elem_type and shape must be None when a ValueInfoProto is passed.");
    }
    return nb::cast<ValueInfoProto>(name_or_proto);
  }
  if (elem_type.is_none()) {
    ValueInfoProto vi;
    vi.set_name(nb::cast<std::string>(name_or_proto));
    if (!doc_string.is_none()) {
      vi.set_doc_string(nb::cast<std::string>(doc_string));
    }
    return vi;
  }
  nb::module_ helper = ImportHelper();
  nb::object py_doc = doc_string.is_none() ? nb::cast(std::string()) : doc_string;
  nb::object built = helper.attr("make_tensor_value_info")(name_or_proto, elem_type, shape,
                                                           nb::arg("doc_string") = py_doc);
  return nb::cast<ValueInfoProto>(built);
}

// Build a NodeProto via helper.make_node (forwards **attrs kwargs) and append
// it to the given repeated node field via the macro-generated ``add_node``
// pointer overload, then return a reference to the stored node.
template <typename ProtoT>
NodeProto &AddNodeImpl(ProtoT &proto, nb::object op_type, nb::object inputs, nb::object outputs,
                       const nb::kwargs &kwargs) {
  nb::module_ helper = ImportHelper();
  nb::object built = helper.attr("make_node")(op_type, inputs, outputs, **kwargs);
  NodeProto *stored = proto.add_node(nb::cast<const NodeProto &>(built));
  return *stored;
}

bool HasBorrowedRawData(const ModelProto &model) {
  if (!model.has_graph()) {
    return false;
  }
  // IteratorTensorProto currently exposes a mutable GraphProto traversal API.
  // The scan is read-only, so cast away constness only to walk the graph.
  auto &mutable_model = const_cast<ModelProto &>(model);
  IteratorTensorProto it(&mutable_model.ref_graph());
  while (it.next()) {
    if (it->ref_raw_data().is_borrowed()) {
      return true;
    }
  }
  return false;
}

// Applies a single ``field=value`` keyword argument to a proto instance,
// mirroring the behavior of ``google.protobuf.Message(**kwargs)``.
// Repeated fields are populated from the provided iterable (python list/tuple
// or another ``RepeatedField``): the field is reset with ``clear()`` and then
// filled with ``extend()``. Every other field (scalar, string, bytes, enum or
// message) is assigned through its regular attribute setter.
void SetProtoFieldFromKwarg(nb::handle py, const std::string &key, nb::handle value) {
  const bool repeated_like = nb::hasattr(value, "__iter__") && !nb::isinstance<nb::str>(value) &&
                             !nb::isinstance<nb::bytes>(value) && !nb::isinstance<Message>(value);
  if (repeated_like) {
    nb::object attr = nb::getattr(py, key.c_str());
    attr.attr("clear")();
    attr.attr("extend")(value);
  } else {
    nb::setattr(py, key.c_str(), value);
  }
}

} // namespace

#define PYDEFINE_PROTO(m, cls)                                                                     \
  nb::class_<cls, Message> nb_##cls(m, #cls, cls::DOC);                                            \
  nb_##cls.def(nb::init<>())

#define PYDEFINE_SUBPROTO(m, cls, subname)                                                         \
  nb::class_<cls::subname, Message> nb_sub_##cls##subname(m, #subname, cls::subname::DOC);         \
  nb_sub_##cls##subname.def(nb::init<>())

#define PYDEFINE_PROTO_WITH_SUBTYPES(m, cls)                                                       \
  nb::class_<cls, Message> nb_##cls(m, #cls, cls::DOC);                                            \
  nb_##cls.def(nb::init<>());

#define PYDEFINE_PROTO_WITH_SUBTYPES2(m, cls, subcls)                                              \
  nb::class_<cls::subcls, Message> nb_sub_##cls##subcls(nb_##cls, #subcls, cls::subcls::DOC);      \
  nb_sub_##cls##subcls.def(nb::init<>());

#define _PYADD_PROTO_SERIALIZATION(cls, name_inst) pyadd_proto_serialization(name_inst);

#define PYADD_PROTO_SERIALIZATION(cls) _PYADD_PROTO_SERIALIZATION(cls, nb_##cls)
#define PYADD_SUBPROTO_SERIALIZATION(cls, sub)                                                     \
  _PYADD_PROTO_SERIALIZATION(cls::sub, nb_sub_##cls##sub)

#define PYFIELD(cls, name)                                                                         \
  def_rw(#name, &cls::name##_, cls::DOC_##name)                                                    \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_REPEATED_STR(cls, name)                                                            \
  def_prop_rw(                                                                                     \
      #name, [](cls &self) -> utils::RepeatedField<utils::String> & { return self.name##_; },      \
      [](cls &self, nb::object obj) {                                                              \
        auto &field = self.name##_;                                                                \
        field.clear();                                                                             \
        if (nb::isinstance<utils::RepeatedField<utils::String>>(obj)) {                            \
          field.extend(nb::cast<utils::RepeatedField<utils::String> &>(obj));                      \
        } else {                                                                                   \
          for (auto it : nb::borrow<nb::iterable>(obj)) {                                          \
            if (nb::isinstance<nb::bytes>(it)) {                                                   \
              nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(it);                               \
              std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());       \
              field.push_back(utils::String(st));                                                  \
            } else {                                                                               \
              field.push_back(utils::String(nb::cast<std::string>(it)));                           \
            }                                                                                      \
          }                                                                                        \
        }                                                                                          \
      },                                                                                           \
      cls::DOC_##name)                                                                             \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_STR(cls, name)                                                                     \
  def_prop_rw(                                                                                     \
      #name, [](const cls &self) -> std::string { return self.ref_##name(); },                     \
      [](cls &self, nb::object obj) {                                                              \
        if (obj.is_none()) {                                                                       \
          self.clear_##name();                                                                     \
        } else if (nb::isinstance<nb::str>(obj)) {                                                 \
          std::string st = nb::cast<std::string>(obj);                                             \
          self.set_##name(st);                                                                     \
        } else if (nb::isinstance<nb::bytes>(obj)) {                                               \
          nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(obj);                                  \
          std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());           \
          self.set_##name(st);                                                                     \
        } else {                                                                                   \
          self.set_##name(nb::cast<cls::name##_t &>(obj));                                         \
        }                                                                                          \
      },                                                                                           \
      cls::DOC_##name, nb::for_setter(nb::arg("value").none()))                                    \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define PYFIELD_STR_AS_BYTES(cls, name)                                                            \
  def_prop_rw(                                                                                     \
      #name,                                                                                       \
      [](const cls &self) -> nb::bytes {                                                           \
        std::string s = self.ref_##name();                                                         \
        return nb::bytes(s.data(), s.size());                                                      \
      },                                                                                           \
      [](cls &self, nb::object obj) {                                                              \
        if (nb::isinstance<nb::str>(obj)) {                                                        \
          std::string st = nb::cast<std::string>(obj);                                             \
          self.set_##name(st);                                                                     \
        } else if (nb::isinstance<nb::bytes>(obj)) {                                               \
          nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(obj);                                  \
          std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());           \
          self.set_##name(st);                                                                     \
        } else {                                                                                   \
          self.set_##name(nb::cast<cls::name##_t &>(obj));                                         \
        }                                                                                          \
      },                                                                                           \
      cls::DOC_##name)                                                                             \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define _PYFIELD_OPTIONAL_CTYPE(cls, name, ctype)                                                  \
  def_prop_rw(                                                                                     \
      #name,                                                                                       \
      [](cls &self) -> nb::object {                                                                \
        if (!self.has_##name())                                                                    \
          return nb::none();                                                                       \
        return nb::cast(self.ref_##name(), nb::rv_policy::reference);                              \
      },                                                                                           \
      [](cls &self, nb::object obj) {                                                              \
        if (obj.is_none()) {                                                                       \
          self.reset_##name();                                                                     \
        } else if (nb::isinstance<nb::ctype##_>(obj)) {                                            \
          self.set_##name(nb::cast<ctype>(obj));                                                   \
        } else {                                                                                   \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");     \
        }                                                                                          \
      },                                                                                           \
      cls::DOC_##name, nb::for_setter(nb::arg("value").none()))                                    \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_INT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, int)
#define PYFIELD_OPTIONAL_FLOAT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, float)

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                          \
  def_prop_rw(                                                                                     \
      #name, [](cls & self)->cls::name##_t * {                                                     \
        if (!self.name##_.has_value()) {                                                           \
          if (self.has_oneof_##name())                                                             \
            return nullptr;                                                                        \
          self.name##_.set_empty_value();                                                          \
        }                                                                                          \
        return &(*self.name##_);                                                                   \
      },                                                                                           \
      [](cls &self, nb::object obj) {                                                              \
        if (obj.is_none()) {                                                                       \
          self.name##_.reset();                                                                    \
        } else if (nb::isinstance<cls::name##_t>(obj)) {                                           \
          self.name##_ = nb::cast<cls::name##_t &>(obj);                                           \
        } else {                                                                                   \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");     \
        }                                                                                          \
      },                                                                                           \
      nb::rv_policy::reference_internal, cls::DOC_##name, nb::for_setter(nb::arg("value").none())) \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")                    \
      .def(                                                                                        \
          "add_" #name, [](cls & self)->cls::name##_t & {                                          \
            self.name##_.set_empty_value();                                                        \
            return *self.name##_;                                                                  \
          },                                                                                       \
          nb::rv_policy::reference_internal, "Sets an empty value.")

#define SHORTEN_CODE(cls, dtype)                                                                   \
  def_prop_ro_static(#dtype, [](nb::handle) -> int { return static_cast<int>(cls::dtype); })

#define DECLARE_REPEATED_FIELD(T, inst_name)                                                       \
  nb::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);

#define DECLARE_REPEATED_FIELD_PROTO(T, inst_name)                                                 \
  nb::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);        \
  nb::class_<utils::RepeatedProtoField<T>> inst_name##_proto(m, "RepeatedProtoField" #T,           \
                                                             "RepeatedProtoField" #T);

#define DECLARE_REPEATED_FIELD_SUBPROTO(cls, T, inst_name)                                         \
  nb::class_<utils::RepeatedField<cls::T>> inst_name(m, "RepeatedField" #cls #T,                   \
                                                     "RepeatedField" #cls #T);                     \
  nb::class_<utils::RepeatedProtoField<cls::T>> inst_name##_proto(m, "RepeatedProtoField" #cls #T, \
                                                                  "RepeatedProtoField" #cls #T);

template <typename cls> void pyadd_proto_serialization(nb::class_<cls, Message> &name_inst) {
  name_inst
      .def(
          "__init__",
          [](cls *self, nb::kwargs kwargs) {
            new (self) cls();
            if (kwargs.size() == 0)
              return;
            // The wrapping python object exists but is not yet flagged as
            // ready during ``__init__``; mark it so the regular attribute
            // getters/setters used below can extract the C++ instance.
            nb::object py = nb::find(*self);
            nb::inst_mark_ready(py);
            for (auto item : kwargs) {
              SetProtoFieldFromKwarg(py, nb::cast<std::string>(item.first), item.second);
            }
          },
          "Creates an instance. Keyword arguments are set as fields, following the "
          "protobuf API, e.g. ``TensorProto(dims=[2, 2], data_type=TensorProto.FLOAT)``.")
      .def(
          "Clear", [](cls &self) { self.CopyFrom(cls()); }, "Clears the object.")
      .def(
          "ParseFromString",
          [](cls &self, nb::bytes data, nb::object options) {
            const uint8_t *bytes_ptr = reinterpret_cast<const uint8_t *>(data.data());
            ONNX_LIGHT_NAMESPACE::utils::StringStream stream(bytes_ptr,
                                                             static_cast<int64_t>(data.size()));
            if (nb::isinstance<ParseOptions &>(options)) {
              ParseOptions &parse_options = nb::cast<ParseOptions &>(options);
              if (parse_options.format == SerializeFormat::kOrtFlatbuffers) {
                // Recursion-OOM guard: validate the depth limit before any
                // parsing begins so that a maliciously crafted .ort file cannot
                // exhaust the call stack once the flatbuffer reader is
                // implemented.
                EXT_ENFORCE(parse_options.max_recursion_depth > 0,
                            "ParseFromString: ParseOptions::max_recursion_depth must be > 0 "
                            "(got ",
                            parse_options.max_recursion_depth,
                            "). "
                            "The ORT flatbuffer parser uses this limit to reject models "
                            "nested more deeply than the configured value, preventing stack "
                            "overflow on adversarially deep inputs.");
                EXT_ENFORCE(parse_options.max_tensor_size_bytes >= 0,
                            "ParseFromString: ParseOptions::max_tensor_size_bytes must be "
                            ">= 0 (got ",
                            parse_options.max_tensor_size_bytes,
                            "). Use 0 to disable the limit or a positive value to cap "
                            "tensor allocations.");
                EXT_THROW("ParseFromString: SerializeFormat::kOrtFlatbuffers is not "
                          "implemented yet. Use SerializeFormat::kOnnx for now.");
              } else {
                EXT_ENFORCE(parse_options.format == SerializeFormat::kOnnx,
                            "ParseFromString: unrecognised SerializeFormat value ",
                            static_cast<int>(parse_options.format));
                if (parse_options.is_parallel()) {
                  stream.StartThreadPool(parse_options.num_threads);
                }
                self.ParseFromStream(stream, parse_options);
                if (parse_options.is_parallel()) {
                  stream.WaitForDelayedBlock();
                }
              }
            } else {
              ParseOptions opts;
              self.ParseFromStream(stream, opts);
            }
          },
          nb::arg("data"), nb::arg("options") = nb::none(),
          "Parses a sequence of bytes to fill this instance.")
      .def(
          "ParseFromString",
          [](cls &self, const std::string &raw, nb::object options) {
            if (nb::isinstance<ParseOptions &>(options)) {
              self.ParseFromString(raw, nb::cast<ParseOptions &>(options));
            } else {
              self.ParseFromString(raw);
            }
          },
          nb::arg("data"), nb::arg("options") = nb::none(),
          "Parses a string to fill this instance.")
      .def(
          "ParseFromFile",
          [](cls &self, const std::string &file_path, nb::object options,
             const std::string &external_data_file) {
            std::unique_ptr<utils::BinaryStream> stream;
            const bool has_opts = nb::isinstance<ParseOptions &>(options);
            const bool wants_no_copy = has_opts && nb::cast<ParseOptions &>(options).no_copy;
            const FileLoadMode mode =
                has_opts ? nb::cast<ParseOptions &>(options).file_load_mode : FileLoadMode::kAuto;
            if (!external_data_file.empty()) {
              // TwoFilesStream reads the main model file through a buffered std::ifstream
              // (it derives from FileStream), while the separate weights file is the large
              // payload. On the no_copy=True path the weights file is memory-mapped once and
              // every tensor borrows a zero-copy view of it (see
              // TwoFilesStream::borrow_weights_bytes). file_load_mode=MMAP requests the same
              // model-owned mmap + zero-copy borrow for the weights file even when no_copy is
              // not set; the borrowed views keep the mapping alive, so this is safe here even
              // though it is not for a single-file model.
              auto two_stream = std::make_unique<utils::TwoFilesStream>(
                  file_path, external_data_file, mode == FileLoadMode::kMmap && !wants_no_copy);
              stream.reset(two_stream.release());
            } else if (mode == FileLoadMode::kMmap) {
              // A single-file model is memory-mapped once and, when no_copy is set, every
              // tensor borrows a zero-copy view into the mapping. This is safe because each
              // borrowed span retains the mmap ownership token (MmapFileStream::zero_copy_owner),
              // so the mapping outlives the stream object destroyed when ParseFromFile returns.
              stream.reset(new utils::MmapFileStream(file_path));
            } else {
              // mode == kFileStream or kAuto: use the buffered FileStream. Memory
              // mapping is not the default for AUTO; MmapFileStream is only used when
              // explicitly requested via file_load_mode=MMAP. FileStream::CanNoCopy() is
              // false, so no_copy=True silently falls back to copying inline raw_data,
              // which keeps borrowed pointers from outliving the stream object below.
              stream.reset(new utils::FileStream(file_path));
            }
            if (nb::isinstance<ParseOptions &>(options)) {
              ParseOptions &coptions = nb::cast<ParseOptions &>(options);
              if (coptions.is_parallel()) {
                stream->StartThreadPool(coptions.num_threads);
              }
              ParseProtoFromStream(self, *stream, coptions);
              if (coptions.is_parallel()) {
                stream->WaitForDelayedBlock();
              }
            } else {
              ParseOptions opts;
              ParseProtoFromStream(self, *stream, opts);
            }
          },
          nb::arg("name"), nb::arg("options") = nb::none(), nb::arg("external_data_file") = "",
          "Parses a binary file to fill this instance.")
      .def(
          "SerializeSize",
          [](cls &self, nb::object options) -> SerializeSizeResult {
            if (nb::isinstance<SerializeOptions &>(options)) {
              utils::StringWriteStream out;
              return self.SerializeSize(out, nb::cast<SerializeOptions &>(options));
            } else {
              return self.SerializeSize();
            }
          },
          nb::arg("options") = nb::none(), "Returns the size once serialized without serializing.")
      .def(
          "SerializeToString",
          [](cls &self, nb::object options) {
            std::string out;
            bool ok = false;
            if (nb::isinstance<SerializeOptions &>(options)) {
              ok = self.SerializeToString(out, nb::cast<SerializeOptions &>(options));
            } else {
              SerializeOptions opts;
              ok = self.SerializeToString(out, opts);
            }
            if (!ok) {
              throw std::runtime_error(
                  "SerializeToString: output exceeded SerializeOptions.max_serialized_size_bytes.");
            }
            return nb::bytes(out.data(), out.size());
          },
          nb::arg("options") = nb::none(), "Serializes this instance into a sequence of bytes.")
      .def(
          "ByteSize", [](cls &self) { return self.SerializeSize().size(); },
          "Returns the serialized size in bytes, following the protobuf API.")
      .def(
          "SerializeToFile",
          [](cls &self, const std::string &file_path, nb::object options,
             std::string &external_data_file) {
            cls *to_write = &self;
            std::optional<ModelProto> owned_copy;
            if constexpr (std::is_same_v<cls, ModelProto>) {
              if (!external_data_file.empty() && HasBorrowedRawData(self)) {
                owned_copy = MakeOwnedModelProtoCopy(self);
                to_write = &(*owned_copy);
              }
            }
            std::unique_ptr<utils::BinaryWriteStream> stream =
                external_data_file.empty()
                    ? std::unique_ptr<utils::BinaryWriteStream>(
                          new utils::FileWriteStream(file_path))
                    : std::unique_ptr<utils::BinaryWriteStream>(
                          new utils::TwoFilesWriteStream(file_path, external_data_file));
            if (nb::isinstance<SerializeOptions &>(options)) {
              if (!SerializeProtoToStream(*to_write, *stream, nb::cast<SerializeOptions &>(options),
                                          !external_data_file.empty())) {
                throw std::runtime_error("SerializeToFile: output exceeded "
                                         "SerializeOptions.max_serialized_size_bytes.");
              }
            } else {
              SerializeOptions opts;
              if (!SerializeProtoToStream(*to_write, *stream, opts, !external_data_file.empty())) {
                throw std::runtime_error("SerializeToFile: output exceeded "
                                         "SerializeOptions.max_serialized_size_bytes.");
              }
            }
          },
          nb::arg("name"), nb::arg("options") = nb::none(), nb::arg("external_data_file") = "",
          "Serializes this instance into a file. If ``external_data_size`` is not empty, big "
          "weights are stored in this (depending on ``options.raw_data_threshold``). "
          "When writing to two files, temporary external-data metadata is cleared so the "
          "in-memory model stays unchanged.")
      .def(
          "SerializeToFileDescriptor",
          [](cls &self, int fd, nb::object options) {
            bool ok = false;
            if (nb::isinstance<SerializeOptions &>(options)) {
              ok = self.SerializeToFileDescriptor(fd, nb::cast<SerializeOptions &>(options));
            } else {
              ok = self.SerializeToFileDescriptor(fd);
            }
            if (!ok) {
              throw std::runtime_error("SerializeToFileDescriptor: output exceeded "
                                       "SerializeOptions.max_serialized_size_bytes.");
            }
          },
          nb::arg("fd"), nb::arg("options") = nb::none(),
          "Serializes this instance into an open file descriptor without closing it.");

  // Helper lambda for ostream serialization, used by both SerializeToOstream and
  // SerializeToOStream to avoid code duplication
  auto serialize_to_ostream_impl = [](cls &self, nb::object output) {
    std::string out;
    SerializeOptions opts;
    bool ok = self.SerializeToString(out, opts);
    if (!ok) {
      throw std::runtime_error("Serialization to ostream failed (likely exceeded size limit).");
    }
    output.attr("write")(nb::bytes(out.data(), out.size()));
  };

  name_inst
      .def("SerializeToOstream", serialize_to_ostream_impl, nb::arg("output"),
           "Serializes this instance to a Python file-like object that has a write() method.")
      .def("SerializeToOStream", serialize_to_ostream_impl, nb::arg("output"),
           "Alias for SerializeToOstream (capital S) for protobuf API compatibility. "
           "Serializes this instance to a Python file-like object that has a write() method.")
      .def(
          "__str__",
          [](cls &self) -> std::string {
            utils::PrintOptions opts;
            std::stringstream ss;
            self.PrintToStringStream(ss, opts);
            return ss.str();
          },
          "Creates a printable string for this class.")
      .def(
          "CopyFrom", [](cls &self, const cls &src) { self.CopyFrom(src); },
          "Copies one instance into this one.")
      .def(
          "ClearField",
          [](nb::handle self, const std::string &field_name) {
            // Mirrors ``google.protobuf.Message.ClearField``: resets a single
            // field to its empty/default state while leaving the others intact.
            nb::object attr = nb::getattr(self, field_name.c_str());
            // Repeated fields are containers (RepeatedField / RepeatedProtoField)
            // exposing a python ``clear`` method; emptying the container clears
            // the field. They are never ``Message`` instances.
            if (!nb::isinstance<Message>(attr) && nb::hasattr(attr, "clear")) {
              attr.attr("clear")();
              return;
            }
            // Optional scalar fields and optional/oneof message fields drop
            // their presence bit when their setter receives ``None``.
            try {
              nb::setattr(self, field_name.c_str(), nb::none());
            } catch (nb::python_error &) {
              // Always-present scalar, string and message fields reject ``None``;
              // reset them to the default value carried by a freshly constructed
              // message of the same type.
              nb::object fresh = nb::cast(cls());
              nb::setattr(self, field_name.c_str(), nb::getattr(fresh, field_name.c_str()));
            }
          },
          nb::arg("field_name"), "Clears the field ``field_name``, following the protobuf API.")
      .def(
          "__eq__",
          [](const cls &self, const cls &other) -> bool {
            SerializeOptions opts1, opts2;
            std::string s1;
            self.SerializeToString(s1, opts1);
            std::string s2;
            other.SerializeToString(s2, opts2);
            return s1 == s2;
          },
          nb::arg("other"), "Compares the serialized strings.");
}

template <typename cls>
std::string proto_repr_with_short_line(cls &self,
                                       size_t max_short_repr_length = MAX_SHORT_REPR_LENGTH) {
  // Bound the representation while it is being produced: ``PrintToStringStream`` stops writing
  // fields once the rendered text reaches ``max_short_repr_length`` and terminates it with an
  // ellipsis, so the result always stays on a single short line.
  utils::PrintOptions opts;
  opts.max_short_repr_length = max_short_repr_length;
  std::stringstream ss;
  self.PrintToStringStream(ss, opts);
  return ss.str();
}

template <typename T> void define_repeated_field_type(nb::class_<utils::RepeatedField<T>> &nbcls) {
  nbcls.def(nb::init<>())
      .def(
          "__init__",
          [](utils::RepeatedField<T> *self, nb::iterable iterable) {
            new (self) utils::RepeatedField<T>();
            if (nb::isinstance<utils::RepeatedField<T>>(iterable)) {
              self->extend(nb::cast<utils::RepeatedField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if constexpr (std::is_same_v<T, utils::String>) {
                  self->push_back(nb::cast<T &>(it));
                } else {
                  self->push_back(nb::cast<T>(it));
                }
              }
            }
          },
          nb::arg("iterable"), "Creates a RepeatedField from an iterable.")
      .def("add", &utils::RepeatedField<T>::add, nb::rv_policy::reference, "Adds an empty element.")
      .def("clear", &utils::RepeatedField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedField<T>::size, "Returns the number of elements.")
      .def(
          "__repr__",
          [](utils::RepeatedField<T> &self) -> std::string {
            nb::list values;
            for (auto &it : self) {
              values.append(nb::cast(it, nb::rv_policy::reference));
            }
            return nb::cast<std::string>(nb::repr(values));
          },
          "Returns a python-like representation for the list of values.")
      .def(
          "__getitem__",
          [](utils::RepeatedField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self[index];
          },
          nb::rv_policy::reference, nb::arg("index"), "Returns the element at position index.")
      .def(
          "__delitem__",
          [](utils::RepeatedField<T> &self, nb::slice slice) {
            auto tup = slice.compute(self.size());
            auto [start, stop, step, slice_length] = tup;
            self.remove_range(start, stop, step);
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedField<T> &self) -> nb::object {
            auto iterator = nb::make_iterator(nb::type<utils::RepeatedField<T>>(), "iterator",
                                              self.begin(), self.end());
            return nb::cast(iterator);
          },
          "Iterates over the elements.")
      .def(
          "__eq__",
          [](utils::RepeatedField<T> &self, nb::list &obj) -> bool {
            // Compare the size first to avoid materializing the container when
            // the lengths already differ.
            if (self.size() != obj.size())
              return false;
            // Materialize the container into a python list and delegate to the
            // python list comparison so element types (``str``/``bytes``/
            // :class:`String` or numbers) compare as expected.
            nb::list values;
            for (auto &it : self) {
              values.append(nb::cast(it, nb::rv_policy::reference));
            }
            return values.equal(obj);
          },
          nb::arg("other"), "Compares the container to a list of values.");
}

template <typename T>
void define_repeated_field_type_extend(nb::class_<utils::RepeatedField<T>> &nbcls) {
  nbcls
      .def(
          "append", [](utils::RepeatedField<T> &self, T v) { self.push_back(v); }, nb::arg("item"),
          "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if constexpr (std::is_same_v<T, utils::String>) {
                  self.push_back(nb::cast<T &>(it));
                } else {
                  self.push_back(nb::cast<T>(it));
                }
              }
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

template <>
void define_repeated_field_type_extend(nb::class_<utils::RepeatedField<utils::String>> &nbcls) {
  nbcls
      .def(
          "append",
          [](utils::RepeatedField<utils::String> &self, nb::handle item) {
            if (nb::isinstance<utils::String>(item)) {
              self.push_back(nb::cast<utils::String &>(item));
            } else if (nb::isinstance<nb::bytes>(item)) {
              nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(item);
              std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
              self.push_back(utils::String(st));
            } else {
              self.push_back(utils::String(nb::cast<std::string>(item)));
            }
          },
          nb::arg("item"),
          "Append one element to the list of values. Accepts ``str``, ``bytes`` or "
          ":class:`String`.")
      .def(
          "extend",
          [](utils::RepeatedField<utils::String> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<utils::String>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<utils::String> &>(iterable));
            } else {
              std::vector<utils::String> values;
              for (auto it : iterable) {
                if (nb::isinstance<utils::String>(it)) {
                  values.push_back(nb::cast<utils::String &>(it));
                } else if (nb::isinstance<nb::bytes>(it)) {
                  nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(it);
                  std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
                  values.push_back(utils::String(st));
                } else {
                  values.emplace_back(utils::String(nb::cast<std::string>(it)));
                }
              }
              self.extend(values);
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

template <typename T>
void define_repeated_field_type_proto(nb::class_<utils::RepeatedField<T>> &nbcls,
                                      nb::class_<utils::RepeatedProtoField<T>> &nbcls_proto) {
  define_repeated_field_type(nbcls);
  nbcls
      .def(
          "append", [](utils::RepeatedField<T> &self, const T &v) { self.push_back(v); },
          nb::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if (nb::isinstance<const T &>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else if (nb::isinstance<T>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          nb::arg("sequence"), "Extends the list of values.")
      .def(
          "add",
          [](utils::RepeatedField<T> &self, nb::kwargs kwargs) -> T & {
            T &element = self.add();
            if (kwargs.size() > 0) {
              nb::object py_element = nb::cast(element, nb::rv_policy::reference);
              for (auto item : kwargs) {
                nb::setattr(py_element, nb::cast<nb::str>(item.first), item.second);
              }
            }
            return element;
          },
          nb::rv_policy::reference,
          "Adds an element with optional keyword arguments set as fields.");
  nbcls_proto.def(nb::init<>())
      .def(
          "add",
          [](utils::RepeatedProtoField<T> &self, nb::kwargs kwargs) -> std::shared_ptr<T> {
            self.add();
            std::shared_ptr<T> element = self.shared_at(self.size() - 1);
            if (kwargs.size() > 0) {
              nb::object py_element = nb::cast(element);
              for (auto item : kwargs) {
                nb::setattr(py_element, nb::cast<nb::str>(item.first), item.second);
              }
            }
            return element;
          },
          "Adds an element. Keyword arguments are set as fields on the new element.")
      .def("clear", &utils::RepeatedProtoField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedProtoField<T>::size, "Returns the number of elements.")
      .def(
          "__repr__",
          [](utils::RepeatedProtoField<T> &self) -> std::string {
            nb::list values;
            for (auto &it : self) {
              values.append(nb::cast(it, nb::rv_policy::reference));
            }
            return nb::cast<std::string>(nb::repr(values));
          },
          "Returns a python-like representation for the list of values.")
      .def(
          "__getitem__",
          [](utils::RepeatedProtoField<T> &self, int index) -> std::shared_ptr<T> {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self.shared_at(static_cast<size_t>(index));
          },
          nb::arg("index"), "Returns the element at position index.")
      .def(
          "__setitem__",
          [](utils::RepeatedProtoField<T> &self, int index, const T &value) {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            self.get(static_cast<size_t>(index)) = std::make_shared<T>(value);
          },
          nb::arg("index"), nb::arg("value"),
          "Replaces the element at position index with a copy of value.")
      .def(
          "__setitem__",
          [](utils::RepeatedProtoField<T> &self, nb::slice slice, nb::iterable values) {
            auto [start, stop, step, slice_length] = slice.compute(self.size());
            // Materialize the replacement elements as shared pointers (copies of
            // the provided values) so they can be spliced into the container.
            std::vector<std::shared_ptr<T>> replacement;
            for (auto it : values) {
              if (nb::isinstance<const T &>(it)) {
                replacement.push_back(std::make_shared<T>(nb::cast<const T &>(it)));
              } else if (nb::isinstance<T>(it)) {
                replacement.push_back(std::make_shared<T>(nb::cast<T>(it)));
              } else {
                EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
              }
            }
            // Extended slices (step != 1) require a replacement of matching length
            // and perform an element-wise assignment, matching Python list semantics.
            if (step != 1) {
              EXT_ENFORCE(replacement.size() == slice_length, "attempt to assign sequence of size ",
                          static_cast<int>(replacement.size()), " to extended slice of size ",
                          static_cast<int>(slice_length));
              Py_ssize_t i = start;
              for (size_t r = 0; r < slice_length; ++r, i += step) {
                self.get(static_cast<size_t>(i)) = replacement[r];
              }
              return;
            }
            // Simple slices (step == 1) may change the container length. Snapshot the
            // existing owning pointers, splice in the replacement, then rebuild the
            // container preserving the identity of the untouched elements.
            std::vector<std::shared_ptr<T>> current;
            current.reserve(self.size());
            for (size_t i = 0; i < self.size(); ++i)
              current.push_back(self.shared_at(i));
            std::vector<std::shared_ptr<T>> result;
            result.reserve(current.size() - slice_length + replacement.size());
            for (Py_ssize_t i = 0; i < start; ++i)
              result.push_back(current[static_cast<size_t>(i)]);
            for (auto &ptr : replacement)
              result.push_back(ptr);
            for (size_t i = static_cast<size_t>(stop); i < current.size(); ++i)
              result.push_back(current[i]);
            self.clear();
            for (auto &ptr : result) {
              self.add();
              self.get(self.size() - 1) = ptr;
            }
          },
          nb::arg("index"), nb::arg("value"),
          "Replaces the elements selected by a slice with copies of the values.")
      .def(
          "__delitem__",
          [](utils::RepeatedProtoField<T> &self, nb::slice slice) {
            auto tup = slice.compute(self.size());
            auto [start, stop, step, slice_length] = tup;
            self.remove_range(start, stop, step);
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedProtoField<T> &self) {
            // Materialize a python list of shared_ptr-backed wrappers so each
            // element keeps the underlying C++ object alive independently of
            // the container (e.g. across ``del container[:]`` after iteration
            // has started).
            nb::list values;
            for (size_t i = 0; i < self.size(); ++i) {
              values.append(nb::cast(self.shared_at(i)));
            }
            return nb::iter(values);
          },
          "Iterates over the elements.")
      .def(
          "__eq__",
          [](utils::RepeatedField<T> &self, nb::list &obj) -> bool {
            if (self.size() != obj.size())
              return false;
            for (size_t i = 0; i < self.size(); ++i) {
              if (!nb::isinstance<T &>(obj[i]))
                return false;
              std::string s1, s2;
              self[i].SerializeToString(s1);
              nb::cast<T &>(obj[i]).SerializeToString(s2);
              if (s1 != s2)
                return false;
            }
            return true;
          },
          "Compares the container to a list of objects.")
      .def(
          "append", [](utils::RepeatedProtoField<T> &self, const T &v) { self.push_back(v); },
          nb::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedProtoField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedProtoField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedProtoField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if (nb::isinstance<const T &>(it)) {
                  self.push_back(nb::cast<const T &>(it));
                } else if (nb::isinstance<T>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

void AddOnnxPyProto(nb::module_ &m) {
  m.doc() = "onnx from python without protobuf but using the same format";
  m.attr("IR_VERSION") = static_cast<int>(IR_VERSION);

  m.def(
      "utils_onnx_read_varint64",
      [](nb::bytes data) -> nb::tuple {
        std::string raw(static_cast<const char *>(data.data()), data.size());
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
        utils::StringStream st(ptr, raw.size());
        int64_t value = st.next_int64();
        return nb::make_tuple(value, st.tell());
      },
      nb::arg("data"),
      R"pbdoc(Reads a int64_t (protobuf format)
:param data: bytes
:return: 2-tuple, value and number of read bytes
)pbdoc");

  m.def(
      "collect_external_inputs",
      [](nb::handle nodes) {
        return WithNodeList(nodes, [](const utils::RepeatedProtoField<NodeProto> &typed_nodes) {
          return core::graph::CollectExternalInputs(typed_nodes);
        });
      },
      nb::arg("nodes"),
      "Returns the list of input names referenced by ``nodes`` that are not "
      "produced as outputs by any node in the same list. The function "
      "recursively inspects names captured by subgraph attributes "
      "(``GRAPH`` / ``GRAPHS``). "
      "The returned list preserves first-seen order and contains no duplicates; "
      "it skips empty input names.");

  m.def(
      "collect_node_inputs",
      [](const NodeProto &node) { return core::graph::CollectNodeInputs(node); }, nb::arg("node"),
      "Returns the full list of tensor / sequence names that a single ``node`` depends on "
      "at runtime. Includes names referenced by ``node.input()`` and external inputs "
      "captured by subgraph attributes (``GRAPH`` / ``GRAPHS``), preserves "
      "first-seen order without duplicates, and skips empty input names.");

  m.def(
      "collect_remaining_inputs",
      [](nb::handle nodes, const std::vector<std::string> &outputs) {
        return WithNodeList(nodes,
                            [&outputs](const utils::RepeatedProtoField<NodeProto> &typed_nodes) {
                              return core::graph::CollectRemainingInputs(typed_nodes, outputs);
                            });
      },
      nb::arg("nodes"), nb::arg("outputs"),
      "Returns, for every node in ``nodes``, the list of input names that must "
      "already be available before that node runs in order to eventually produce "
      "the requested ``outputs``. Starting from ``outputs``, a backward "
      "reachability analysis determines their ancestors; for index ``i`` only the "
      "nodes of ``nodes[i:]`` that contribute to ``outputs`` are kept (unrelated "
      "branches are pruned) and the names they read (including names captured by "
      "subgraph attributes ``GRAPH`` / ``GRAPHS``) that are not produced within "
      "that suffix are reported. ``nodes`` is expected to be in topological "
      "order. The result is a list with one entry per node; each entry preserves "
      "first-seen order, contains no duplicates and skips empty input names.");

  nb::enum_<FileLoadMode>(m, "FileLoadMode",
                          "Selects the file-backed stream implementation used when parsing "
                          "a model from a file path.")
      .value("AUTO", FileLoadMode::kAuto,
             "Pick a compatible stream without memory-mapping (currently the buffered "
             "FileStream; this choice may change in the future).")
      .value("MMAP", FileLoadMode::kMmap, "Force MmapFileStream (memory-mapped file).")
      .value("IFSTREAM", FileLoadMode::kFileStream, "Force FileStream (buffered std::ifstream).");

  nb::enum_<SerializeFormat>(m, "SerializeFormat",
                             "Selects the on-disk serialization format used when parsing or "
                             "serializing a ModelProto.")
      .value("ONNX", SerializeFormat::kOnnx, "Default ONNX protobuf wire format.")
      .value("ORT_FLATBUFFERS", SerializeFormat::kOrtFlatbuffers,
             "Flatbuffer-based format used by onnxruntime (``.ort`` files). "
             "Not implemented yet; setting this format raises an error when "
             "parsing or serializing.");

  nb::class_<TensorBufferOptions>(m, "TensorBufferOptions",
                                  "Common options for tensor buffer operations: in-place "
                                  "consolidation, serialization, and parsing.")
      .def(nb::init<>())
      .def_rw("raw_data_threshold", &TensorBufferOptions::raw_data_threshold,
              "Minimum raw_data size (in bytes) to include in buffer operations. "
              "Tensors whose raw_data is smaller than this threshold are left in-place. "
              "Default: 64 (kSmallTensorDataThresholdBytes).")
      .def_rw("alignment", &TensorBufferOptions::alignment,
              "If > 0, each tensor's offset within the buffer is padded to a multiple of this many "
              "bytes. 0 disables alignment. Use 4096 for mmap-friendly page-aligned offsets.");

  nb::class_<RawDataCallback>(
      m, "RawDataCallback",
      "Reusable :attr:`ParseOptions.raw_data_callback` that keeps the default C++ allocation "
      "(tensor ownership unchanged) while letting users observe parsing progress.\n\n"
      "It is callable as ``fn(tensor, graph)``: it forwards the freshly parsed "
      ":class:`TensorProto` "
      "to "
      "the optional ``on_tensor`` callable (for example to print progress) and always returns "
      "``None``, so the tensor's ``raw_data`` is left to the default allocator. Assign an "
      "instance to :attr:`ParseOptions.raw_data_callback`, or subclass it and override "
      "``__call__`` for richer behavior.")
      .def(nb::init<nb::object>(), nb::arg("on_tensor").none() = nb::none(),
           "Builds the callback. ``on_tensor`` is an optional callable invoked as "
           "``on_tensor(tensor)`` for every parsed tensor; pass ``None`` (the default) for a "
           "no-op that simply preserves the default allocation.")
      .def_prop_rw(
          "on_tensor", [](RawDataCallback &self) -> nb::object { return self.on_tensor; },
          [](RawDataCallback &self, nb::object value) { self.on_tensor = value; },
          "Optional callable invoked as ``on_tensor(tensor)`` for every parsed tensor; "
          "``None`` disables it.",
          nb::for_setter(nb::arg("value").none()))
      .def(
          "__call__",
          [](RawDataCallback &self, TensorProto &tensor, nb::object /*graph*/) -> nb::object {
            if (self.on_tensor.is_valid() && !self.on_tensor.is_none()) {
              self.on_tensor(nb::cast(&tensor, nb::rv_policy::reference));
            }
            return nb::none();
          },
          nb::arg("tensor"), nb::arg("graph").none() = nb::none(),
          "Invokes ``on_tensor(tensor)`` when set and returns ``None`` so the tensor's "
          "``raw_data`` keeps the default C++ allocation. ``graph`` is the tensor's parent "
          "GraphProto (or ``None``) and is accepted but ignored.");

  nb::class_<ParseOptions, TensorBufferOptions>(m, "ParseOptions",
                                                "Parsing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &ParseOptions::skip_raw_data,
              "if true, raw data will not be read but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw("num_threads", &ParseOptions::num_threads,
              "Number of threads to use for parallel reading. Any negative value "
              "(``-1`` is the default) picks a sensible value based on the number "
              "of available CPU cores. ``1`` means no parallelization, and ``> 1`` "
              "uses exactly that many worker threads.")
      .def("is_parallel", &ParseOptions::is_parallel,
           "Returns True when parallel reading should be enabled (num_threads != 1).")
      .def_rw("min_parallel_block_size", &ParseOptions::min_parallel_block_size,
              "minimum raw-data block size in bytes to submit to the thread pool when parallel "
              "reading is enabled (num_threads != 1); "
              "blocks smaller than this value are read on the main thread to avoid thread-pool "
              "overhead")
      .def_rw("no_copy", &ParseOptions::no_copy,
              "If true, raw_data bytes are not copied during parsing. Inline protobuf raw_data "
              "borrows directly from the source bytes buffer, so the caller MUST keep the "
              "original bytes object alive for as long as the parsed model is in use. For "
              "external-data files, each weights file is loaded once into a shared model-owned "
              "buffer and every tensor borrows a view into that buffer.")
      .def_rw("_touch_raw_data_pages", &ParseOptions::_touch_raw_data_pages,
              "If true, this option touches one byte per page in every non-empty tensor "
              "raw_data buffer (plus the last byte) after parsing, forcing lazy page faults "
              "to occur during parse timing.")
      .def_rw("tiny_external_data_threshold", &ParseOptions::tiny_external_data_threshold,
              "Loads tiny external-data tensors inline during parsing when reading from a model "
              "file path without an explicit external_data_file stream. Negative disables it "
              "(default); non-negative values load tensors whose declared external-data "
              "``length``/``size`` is strictly below the threshold (in bytes).")
      .def_rw("file_load_mode", &ParseOptions::file_load_mode,
              "Selects the file-backed stream used when parsing a model from a path: "
              "FileLoadMode.AUTO (default) picks a compatible stream and does not "
              "memory-map (currently the buffered std::ifstream-based FileStream, but "
              "this choice may change in the future), FileLoadMode.MMAP forces "
              "MmapFileStream, and FileLoadMode.IFSTREAM always forces the buffered "
              "std::ifstream-based FileStream. "
              "Ignored when parsing from bytes or when an external_data_file is provided.")
      .def_rw("format", &ParseOptions::format,
              "Selects the on-disk serialization format expected when parsing. "
              "SerializeFormat.ONNX (default) parses the ONNX protobuf wire format; "
              "SerializeFormat.ORT_FLATBUFFERS parses the onnxruntime flatbuffer format "
              "(``.ort`` files). The flatbuffer path is not implemented yet and raises "
              "an error when used.")
      .def_rw("max_recursion_depth", &ParseOptions::max_recursion_depth,
              "Maximum nesting depth of protobuf sub-messages accepted while parsing "
              "(default 100). Protects against stack overflow / out-of-memory from deeply "
              "nested messages; parsing raises an error when a message nests deeper than "
              "this value. The default matches protobuf's own limit of 100 so that any "
              "model protobuf accepts is also accepted here, including deeply nested "
              "control-flow models with dozens of nested Loop/If subgraphs.")
      .def_rw("max_tensor_size_bytes", &ParseOptions::max_tensor_size_bytes,
              "Maximum number of bytes that may be allocated for a single tensor's raw "
              "data (or packed repeated-field payload) during parsing (default 0 = no limit). "
              "Protects against OOM from maliciously or accidentally large size prefixes in "
              "the wire format: parsing raises an error when the declared byte count for any "
              "single tensor allocation exceeds this value. The check fires before the "
              "allocation. Set to 0 to disable (default) or to a value comfortably above the "
              "largest legitimate tensor you expect, e.g. 2 * 1024 ** 3 for a 2 GB cap.")
      .def_prop_rw(
          "raw_data_callback",
          [](ParseOptions &options) -> nb::object {
            if (!options.raw_data_callback) {
              return nb::none();
            }
            if (const PyRawDataCallback *cb =
                    options.raw_data_callback.target<PyRawDataCallback>()) {
              return cb->fn;
            }
            return nb::none();
          },
          [](ParseOptions &options, nb::object fn) {
            if (fn.is_none()) {
              options.raw_data_callback = {};
            } else {
              options.raw_data_callback = PyRawDataCallback{fn};
            }
          },
          "Optional callable invoked for every parsed TensorProto once its ``raw_data`` has "
          "been resolved (including external-data tensors). It is called as "
          "``fn(tensor, graph)`` with the freshly parsed :class:`TensorProto` and its parent "
          ":class:`GraphProto` (or ``None`` when the tensor is parsed on its own) and must return "
          "either ``None`` (ownership unchanged) or a zero-argument callable used as the deleter "
          "attached to the tensor's ``raw_data``; the deleter runs once when that raw_data is "
          "released. Setting it to ``None`` (the default) disables the callback. See "
          ":class:`RawDataCallback` for a ready-made object that only reports progress while "
          "keeping the default allocation.",
          nb::for_setter(nb::arg("value").none()))
      .def_prop_rw(
          "node_callback",
          [](ParseOptions &options) -> nb::object {
            if (!options.node_callback) {
              return nb::none();
            }
            if (const PyNodeCallback *cb = options.node_callback.target<PyNodeCallback>()) {
              return cb->fn;
            }
            return nb::none();
          },
          [](ParseOptions &options, nb::object fn) {
            if (fn.is_none()) {
              options.node_callback = {};
            } else {
              options.node_callback = PyNodeCallback{fn};
            }
          },
          "Optional callable invoked as ``fn(node, graph)`` for every :class:`NodeProto` once it "
          "has been fully parsed. The node and its parent :class:`GraphProto` are passed by "
          "reference; the node may be inspected or edited in place. Setting it to ``None`` (the "
          "default) disables the callback.",
          nb::for_setter(nb::arg("value").none()));

  nb::class_<SerializeOptions, TensorBufferOptions>(m, "SerializeOptions",
                                                    "Serializing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &SerializeOptions::skip_raw_data,
              "if true, raw data will not be written but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw("num_threads", &SerializeOptions::num_threads,
              "Number of threads to use for parallel writing. Any negative value "
              "(``-1`` is the default) picks a sensible value based on the number "
              "of available CPU cores. ``1`` means no parallelization, and ``> 1`` "
              "uses exactly that many worker threads.")
      .def("is_parallel", &SerializeOptions::is_parallel,
           "Returns True when parallel writing should be enabled (num_threads != 1).")
      .def_rw("min_parallel_block_size", &SerializeOptions::min_parallel_block_size,
              "minimum raw-data block size in bytes to submit to the thread pool when parallel "
              "writing is enabled (num_threads != 1); "
              "blocks smaller than this value are written on the main thread to avoid thread-pool "
              "overhead")
      .def_rw("use_external_data_location", &SerializeOptions::use_external_data_location,
              "if true, tensors already marked as EXTERNAL are written to the file specified by "
              "external_data.location; this allows serialization into one or more weights files.")
      .def_rw("max_serialized_size_bytes", &SerializeOptions::max_serialized_size_bytes,
              "Maximum serialized size in bytes allowed for one serialization operation "
              "(default 0 = no limit). The limit applies to the total output size "
              "(protobuf payload + external data). Serialization reports failure when "
              "the computed size exceeds this value.")
      .def_rw("max_external_file_size", &SerializeOptions::max_external_file_size,
              "maximum size in bytes for one external weights file when writing external data; "
              "0 means no limit")
      .def_rw("format", &SerializeOptions::format,
              "Selects the on-disk serialization format produced when serializing. "
              "SerializeFormat.ONNX (default) writes the ONNX protobuf wire format; "
              "SerializeFormat.ORT_FLATBUFFERS writes the onnxruntime flatbuffer format "
              "(``.ort`` files). The flatbuffer path is not implemented yet and raises "
              "an error when used.")
      .def_prop_rw(
          "raw_data_callback",
          [](SerializeOptions &options) -> nb::object {
            if (!options.raw_data_callback) {
              return nb::none();
            }
            if (const PySerializeRawDataCallback *cb =
                    options.raw_data_callback.target<PySerializeRawDataCallback>()) {
              return cb->fn;
            }
            return nb::none();
          },
          [](SerializeOptions &options, nb::object fn) {
            if (fn.is_none()) {
              options.raw_data_callback = {};
            } else {
              options.raw_data_callback = PySerializeRawDataCallback{fn};
            }
          },
          "Optional callable invoked as ``fn(tensor, graph, buffer, size_only)`` for every "
          "tensor carrying ``raw_data`` immediately before serialization. ``graph`` is the "
          "tensor's parent :class:`GraphProto` (or ``None``). The callback "
          "is first called with ``buffer=None`` and ``size_only=True`` and must return "
          "the number of bytes it will serialize. onnx-light then allocates a writable "
          "1-D uint8 buffer of that size and calls ``fn(tensor, graph, buffer, "
          "size_only=False)``; "
          "the callback may update the :class:`TensorProto` metadata in place, must fill "
          "the provided buffer, and must return the same size again. When a tensor was "
          "previously marked as EXTERNAL and still carries ``raw_data`` (for example "
          "after ``tensor.load_external_data(...)``), serialization refreshes that "
          "metadata after the callback so the written ``length`` and ``offset`` match "
          "the rewritten bytes. Setting it to ``None`` (the default) disables the "
          "callback.",
          nb::for_setter(nb::arg("value").none()))
      .def_prop_rw(
          "node_callback",
          [](SerializeOptions &options) -> nb::object {
            if (!options.node_callback) {
              return nb::none();
            }
            if (const PyNodeCallback *cb = options.node_callback.target<PyNodeCallback>()) {
              return cb->fn;
            }
            return nb::none();
          },
          [](SerializeOptions &options, nb::object fn) {
            if (fn.is_none()) {
              options.node_callback = {};
            } else {
              options.node_callback = PyNodeCallback{fn};
            }
          },
          "Optional callable invoked as ``fn(node, graph)`` for every :class:`NodeProto` "
          "immediately before it is serialized. The node and its parent :class:`GraphProto` are "
          "passed by reference (from a working copy of the model, so edits never alter the "
          "caller's model) and the node may be inspected or edited in place. Setting it to "
          "``None`` (the default) disables the callback.",
          nb::for_setter(nb::arg("value").none()));

  nb::class_<SerializeSizeResult>(m, "SerializeSizeResult",
                                  "Splits serialized bytes between proto data and tensor content.")
      .def(nb::init<>())
      .def_rw("small_data_size", &SerializeSizeResult::small_data_size,
              "Bytes written outside the main proto for small tensor payloads.")
      .def_rw("big_data_size", &SerializeSizeResult::big_data_size,
              "Bytes written outside the main proto for big tensor payloads.")
      .def_prop_ro(
          "data_size",
          [](const SerializeSizeResult &self) { return self.small_data_size + self.big_data_size; },
          "Returns the total bytes written outside the main proto.")
      .def_rw("proto_size", &SerializeSizeResult::proto_size,
              "Bytes written into the main protobuf payload.")
      .def("size", &SerializeSizeResult::size, "Returns the total number of serialized bytes.");

  m.def(
      "consolidate_tensors_to_buffer",
      [](ModelProto &model, nb::object opts) {
        if (nb::isinstance<TensorBufferOptions &>(opts)) {
          TensorBufferOptions &copts = nb::cast<TensorBufferOptions &>(opts);
          ConsolidateTensorsToBuffer(model, copts);
        } else {
          TensorBufferOptions default_opts;
          ConsolidateTensorsToBuffer(model, default_opts);
        }
      },
      nb::arg("model"), nb::arg("opts") = nb::none(),
      R"pbdoc(Moves all tensor raw_data (whose size >= opts.raw_data_threshold) into a
single contiguous buffer owned via shared ownership, then rebinds each qualifying
tensor to borrow from that buffer.  After the call the buffer lifetime is managed
by the tensors themselves; the caller does not need to keep any extra reference.

This mirrors the no-copy external-data loading scenario and avoids per-tensor
memory allocations.

:param model: ModelProto whose tensors will be consolidated in-place.
:param opts: TensorBufferOptions (or a subclass such as SerializeOptions) that
    controls the size threshold and byte alignment.  Pass None to use default
    options (threshold=0, alignment=0 → consolidate all tensors without alignment).
)pbdoc");

  m.def(
      "align_external_data_streaming",
      [](const std::string &src_onnx_path, const std::string &dst_onnx_path,
         const std::string &dst_weights_path, int64_t alignment, int64_t chunk_size) -> int64_t {
        return static_cast<int64_t>(AlignExternalDataStreaming(
            src_onnx_path, dst_onnx_path, dst_weights_path, alignment, chunk_size));
      },
      nb::arg("src_onnx_path"), nb::arg("dst_onnx_path"), nb::arg("dst_weights_path"),
      nb::arg("alignment") = 4096, nb::arg("chunk_size") = 4 * 1024 * 1024,
      R"pbdoc(Rewrites an existing two-file ONNX model (``.onnx`` + one or more
external weights file(s)) into a new ``(dst_onnx_path, dst_weights_path)`` pair
so that every tensor's offset inside the destination weights file is aligned to
``alignment`` bytes — without ever loading the full set of weights in memory.
The destination always uses a single consolidated weights file even when the
source spreads tensors across multiple ``external_data.location`` files.

The source ``.onnx`` is parsed with ``skip_raw_data=True``, so only the
initializer metadata (including ``external_data``) is read.  For each tensor
referenced as external data, ``length`` bytes are streamed from the source
weights file to the destination weights file in chunks of ``chunk_size`` bytes,
and the proto's ``external_data`` entries are updated in place to point at the
new file and aligned offset.  Finally, the updated proto is written to
``dst_onnx_path``.

Peak heap usage is therefore bounded by the proto metadata size plus
``chunk_size`` bytes — independent of the total weights size.  Use this when
you need to align weights for mmap or accelerator-friendly loading but cannot
afford to load the full model in memory.

:param src_onnx_path: Path to the source ``.onnx`` file.
    ``external_data.location`` entries are resolved relative to its parent
    directory.
:param dst_onnx_path: Destination ``.onnx`` file (created/truncated).
:param dst_weights_path: Destination weights file (created/truncated).
:param alignment: Alignment in bytes (power of two, >= 1).  Defaults to 4096.
:param chunk_size: Maximum bytes copied per I/O call.  Defaults to 4 MiB.
:return: Total bytes written to ``dst_weights_path`` (including padding).
)pbdoc");

  m.def(
      "save_model_with_shared_external_data",
      [](ModelProto &model, const std::string &dst_onnx_path,
         const SerializeOptions &options) -> int64_t {
        return static_cast<int64_t>(SaveModelWithSharedExternalData(model, dst_onnx_path, options));
      },
      nb::arg("model"), nb::arg("dst_onnx_path"), nb::arg("options") = SerializeOptions{},
      R"pbdoc(Saves a model while reusing already-external weights of any
previously saved model the initializers were taken from.

This is the companion of :func:`align_external_data_streaming` for the scenario
where a first model has already been written to disk (one ``.onnx`` + one or
more weights files) and was then loaded **without** external data, so its
initializers still carry the original ``external_data`` metadata.  A model is
then built that mixes some of those reused initializers with new initializers
carrying inline ``raw_data``.

Calling this function serializes that model to ``dst_onnx_path`` such that:

* Initializers already marked as EXTERNAL are written out as-is: their
  ``external_data`` entries (``location``, ``offset``, ``length``) are kept
  unchanged, so they keep referencing whatever weights file they already
  pointed at.  No byte is copied from those files.  The caller is responsible
  for the recorded ``location`` remaining resolvable relative to
  ``dst_onnx_path``'s parent directory.
* New initializers carrying inline ``raw_data`` are written to a single
  secondary weights file named ``<dst_onnx_path>.data`` (placed next to
  ``dst_onnx_path``) at aligned offsets.  Their inline bytes are then cleared
  from the in-memory proto and their ``external_data`` entries are updated to
  point at that secondary file.  The secondary file therefore contains *only*
  the new weights, and is not created at all when every initializer is reused.

:param model: ModelProto, mutated in place.  After the call, new initializers
    reference the secondary weights file instead of carrying inline
    ``raw_data``.
:param dst_onnx_path: Destination ``.onnx`` file (created/truncated).  The
    secondary weights file (when needed) is created at
    ``dst_onnx_path + ".data"``.
:param options: :class:`SerializeOptions`.  Only ``alignment`` (inherited from
    :class:`TensorBufferOptions`) is honored: it controls the alignment in
    bytes applied to each new tensor's offset in the secondary weights file
    (``0`` disables alignment; use ``4096`` for mmap-friendly pages).  Defaults
    to a freshly constructed :class:`SerializeOptions`.
:return: Total bytes written to the secondary weights file (including any
    alignment padding; ``0`` when no new initializer needed to be written).
)pbdoc");

  m.def(
      "convert_model_to_external_data",
      [](ModelProto &model, bool all_tensors_to_one_file, const std::string &location,
         int64_t size_threshold, bool convert_attribute) {
        EXT_ENFORCE_INVALID(size_threshold >= 0, "size_threshold must be non-negative, got ",
                            size_threshold, ".");
        try {
          ConvertModelToExternalData(model, all_tensors_to_one_file, location,
                                     static_cast<size_t>(size_threshold), convert_attribute);
        } catch (const ExternalDataLocationExistsError &e) {
          PyErr_SetString(PyExc_FileExistsError, e.what());
          throw nb::python_error();
        }
      },
      nb::arg("model"), nb::arg("all_tensors_to_one_file") = true,
      nb::arg("location") = std::string(), nb::arg("size_threshold") = 1024,
      nb::arg("convert_attribute") = false,
      R"pbdoc(Marks every initializer tensor of *model* whose ``raw_data`` is at
least ``size_threshold`` bytes long as EXTERNAL.  The actual bytes are not
written; they remain in ``raw_data`` and are flushed to disk by the next
serialization call (e.g. :func:`onnx_light.onnx.save`).

Mirrors :func:`onnx.external_data_helper.convert_model_to_external_data` on top
of onnx-light's protos.

:param model: ModelProto to modify in place.
:param all_tensors_to_one_file: When True (default), every qualifying tensor
    points at the same external file (``location`` or a generated
    ``<uuid>.data`` name).  When False, each tensor is given its own file
    named after the tensor.
:param location: Relative path of the external data file.  Must be relative to
    the model file.  Ignored when ``all_tensors_to_one_file=False``.  Empty
    means "generate a name".
:param size_threshold: Only tensors whose ``raw_data`` size is greater than or
    equal to ``size_threshold`` bytes are moved to external storage.  Set to
    ``0`` to externalize every tensor with raw data.
:param convert_attribute: When True, also externalize tensors stored inside
    node attributes (``AttributeProto.t`` and ``AttributeProto.tensors``).
:raises ValueError: When ``location`` is an absolute path.
:raises FileExistsError: When ``location`` already exists on disk.
)pbdoc");

  m.def(
      "load_external_data_for_model",
      [](ModelProto &model, const std::string &base_dir) {
        LoadExternalDataForModel(model, base_dir);
      },
      nb::arg("model"), nb::arg("base_dir"),
      R"pbdoc(Loads external tensor bytes into *model* in place.

For every tensor whose data lives in an external file, reads the bytes from
``base_dir`` into the tensor's ``raw_data`` and resets ``data_location`` to
``DEFAULT``, dropping the ``external_data`` entries.

Mirrors :func:`onnx.external_data_helper.load_external_data_for_model`.

:param model: ModelProto whose external tensors are loaded in place.
:param base_dir: Directory that contains the external data files referenced
    by the tensors.
)pbdoc");

  nb::class_<utils::PrintOptions>(m, "PrintOptions", "Printing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &utils::PrintOptions::skip_raw_data,
              "if true, raw data will not be printed but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw("raw_data_threshold", &utils::PrintOptions::raw_data_threshold,
              "if skip_raw_data is true, raw data will be printed only if it is larger than the "
              "threshold")
      .def_rw("inline_threshold", &utils::PrintOptions::inline_threshold,
              "repeated fields with at most this many elements are printed as a bracketed list; "
              "all output is flat (no newlines)")
      .def_rw("max_short_repr_length", &utils::PrintOptions::max_short_repr_length,
              "maximum number of characters produced when printing a short representation; "
              "printing stops once this bound is reached and the text ends with an ellipsis; "
              "zero (the default) means no limit");

  nb::class_<utils::String>(m, "String", "Simplified string with no final null character.")
      .def(nb::init<std::string>())
      .def(
          "__str__", [](const utils::String &self) -> std::string { return self; },
          "Converts this instance into a python string.")
      .def(
          "__add__",
          [](const utils::String &self, const nb::str &s) -> std::string {
            return self + nb::cast<std::string>(s);
          },
          "Concatenates this string and a python string.", nb::is_operator())
      .def(
          "__radd__",
          [](const utils::String &self, const nb::str &s) -> std::string {
            return nb::cast<std::string>(s) + self;
          },
          "Concatenates a python string and this string.", nb::is_operator())
      .def(
          "__repr__", [](const utils::String &self) -> std::string { return "'" + self + "'"; },
          "Representation with surrounding quotes.")
      .def(
          "__len__", [](const utils::String &self) -> int { return self.size(); },
          "Returns the length of the string.")
      .def(
          "__eq__",
          [](const utils::String &self, const std::string &s) -> bool { return self == s; },
          "Compares two strings.")
      .def(
          "__eq__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> bool {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self == st;
          },
          "Compares to a byte string.")
      .def(
          "__eq__",
          [](const utils::String &self, const utils::String &s) -> bool { return self == s; },
          "Compares two String instances.", nb::is_operator())
      .def(
          "__eq__", [](const utils::String &, nb::object) -> bool { return false; },
          nb::arg("other").none(), "Returns False when compared to an object that is not a string.",
          nb::is_operator())
      .def(
          "__ne__",
          [](const utils::String &self, const std::string &s) -> bool { return self != s; },
          "Checks inequality with a python string.", nb::is_operator())
      .def(
          "__ne__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> bool {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self != st;
          },
          "Checks inequality with a byte string.", nb::is_operator())
      .def(
          "__ne__",
          [](const utils::String &self, const utils::String &s) -> bool { return self != s; },
          "Checks inequality with a String.", nb::is_operator())
      .def(
          "__ne__", [](const utils::String &, nb::object) -> bool { return true; },
          nb::arg("other").none(), "Returns True when compared to an object that is not a string.",
          nb::is_operator())
      .def(
          "__lt__",
          [](const utils::String &self, const std::string &s) -> bool { return self < s; },
          "Checks whether this string is less than a python string.", nb::is_operator())
      .def(
          "__lt__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> bool {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self < st;
          },
          "Checks whether this string is less than a byte string.", nb::is_operator())
      .def(
          "__lt__",
          [](const utils::String &self, const utils::String &s) -> bool { return self < s; },
          "Checks whether this string is less than a String.", nb::is_operator())
      .def(
          "__gt__",
          [](const utils::String &self, const std::string &s) -> bool { return self > s; },
          "Checks whether this string is greater than a python string.", nb::is_operator())
      .def(
          "__gt__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> bool {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self > st;
          },
          "Checks whether this string is greater than a byte string.", nb::is_operator())
      .def(
          "__gt__",
          [](const utils::String &self, const utils::String &s) -> bool { return self > s; },
          "Checks whether this string is greater than a String.", nb::is_operator())
      .def(
          "__hash__",
          [](const utils::String &self) -> Py_hash_t {
            nb::str py_str(self.data(), self.size());
            return PyObject_Hash(py_str.ptr());
          },
          "Returns the same hash as the equivalent Python str, enabling use as dict keys.")
      .def(
          "decode",
          [](const utils::String &self, const char *encoding, const char *errors) -> nb::object {
            std::string s = self;
            nb::bytes data(s.data(), s.size());
            return data.attr("decode")(encoding, errors);
          },
          nb::arg("encoding") = "utf-8", nb::arg("errors") = "strict",
          "Decodes the string like a Python :class:`bytes` object, returning a Python str.");

  DECLARE_REPEATED_FIELD(int64_t, rep_int64_t);
  define_repeated_field_type(rep_int64_t);
  define_repeated_field_type_extend(rep_int64_t);

  DECLARE_REPEATED_FIELD(int32_t, rep_int32_t);
  define_repeated_field_type(rep_int32_t);
  define_repeated_field_type_extend(rep_int32_t);

  DECLARE_REPEATED_FIELD(uint64_t, rep_uint64_t);
  define_repeated_field_type(rep_uint64_t);
  define_repeated_field_type_extend(rep_uint64_t);

  DECLARE_REPEATED_FIELD(float, rep_float);
  define_repeated_field_type(rep_float);
  define_repeated_field_type_extend(rep_float);

  DECLARE_REPEATED_FIELD(double, rep_double);
  define_repeated_field_type(rep_double);
  define_repeated_field_type_extend(rep_double);

  nb::class_<utils::RepeatedField<utils::String>> rep_string(m, "RepeatedFieldString",
                                                             "RepeatedFieldString");
  define_repeated_field_type(rep_string);
  // Override __init__ for String to handle str/bytes/String types
  rep_string.def(
      "__init__",
      [](utils::RepeatedField<utils::String> *self, nb::iterable iterable) {
        new (self) utils::RepeatedField<utils::String>();
        if (nb::isinstance<utils::RepeatedField<utils::String>>(iterable)) {
          self->extend(nb::cast<utils::RepeatedField<utils::String> &>(iterable));
        } else {
          for (auto it : iterable) {
            if (nb::isinstance<utils::String>(it)) {
              self->push_back(nb::cast<utils::String &>(it));
            } else if (nb::isinstance<nb::bytes>(it)) {
              nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(it);
              std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
              self->push_back(utils::String(st));
            } else {
              self->push_back(utils::String(nb::cast<std::string>(it)));
            }
          }
        }
      },
      nb::arg("iterable"), "Creates a RepeatedFieldString from an iterable.");
  define_repeated_field_type_extend(rep_string);
  nb::class_<utils::RepeatedStringField, utils::RepeatedField<utils::String>> rep_string_name(
      m, "RepeatedFieldStringName", "RepeatedFieldStringName");
  rep_string_name.def(
      "__iter__",
      [](utils::RepeatedStringField &self) -> nb::object {
        nb::list values;
        for (const auto &it : self) {
          values.append(nb::cast(std::string(it)));
        }
        return nb::iter(values);
      },
      "Iterates over name-like string elements as python str.");

  nb::enum_<OperatorStatus>(m, "OperatorStatus", nb::is_arithmetic())
      .value("EXPERIMENTAL", OperatorStatus::EXPERIMENTAL)
      .value("STABLE", OperatorStatus::STABLE)
      .export_values();

  nb::class_<Message>(m, "Message", "Message, base class for all onnx classes").def(nb::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value);
  PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  nb_StringStringEntryProto.def(
      "HasField",
      [](const StringStringEntryProto &self, const std::string &field_name) -> bool {
        if (field_name == "key")
          return self.has_key();
        if (field_name == "value")
          return self.has_value();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(StringStringEntryProto, rep_ssentry);
  define_repeated_field_type_proto(rep_ssentry, rep_ssentry_proto);

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version);
  PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  nb_OperatorSetIdProto.def(
      "HasField",
      [](const OperatorSetIdProto &self, const std::string &field_name) -> bool {
        if (field_name == "domain")
          return self.has_domain();
        if (field_name == "version")
          return self.has_version();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  nb_OperatorSetIdProto.def(
      "__repr__", [](OperatorSetIdProto &self) { return proto_repr_with_short_line(self); });
  DECLARE_REPEATED_FIELD_PROTO(OperatorSetIdProto, rep_osp);
  define_repeated_field_type_proto(rep_osp, rep_osp_proto);

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names);
  PYADD_PROTO_SERIALIZATION(TensorAnnotation);
  nb_TensorAnnotation.def(
      "HasField",
      [](const TensorAnnotation &self, const std::string &field_name) -> bool {
        if (field_name == "tensor_name")
          return self.has_tensor_name();
        if (field_name == "quant_parameter_tensor_names")
          return self.has_quant_parameter_tensor_names();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(TensorAnnotation, rep_ta);
  define_repeated_field_type_proto(rep_ta, rep_ta_proto);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value);
  PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  nb_IntIntListEntryProto.def(
      "HasField",
      [](const IntIntListEntryProto &self, const std::string &field_name) -> bool {
        if (field_name == "key")
          return self.has_key();
        if (field_name == "value")
          return self.has_value();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(IntIntListEntryProto, rep_iil);
  define_repeated_field_type_proto(rep_iil, rep_iil_proto);

  PYDEFINE_PROTO(m, DeviceConfigurationProto)
      .PYFIELD_STR(DeviceConfigurationProto, name)
      .PYFIELD(DeviceConfigurationProto, num_devices)
      .PYFIELD(DeviceConfigurationProto, device);
  PYADD_PROTO_SERIALIZATION(DeviceConfigurationProto);
  nb_DeviceConfigurationProto.def(
      "HasField",
      [](const DeviceConfigurationProto &self, const std::string &field_name) -> bool {
        if (field_name == "name")
          return self.has_name();
        if (field_name == "num_devices")
          return self.has_num_devices();
        if (field_name == "device")
          return self.has_device();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(DeviceConfigurationProto, rep_dcp);
  define_repeated_field_type_proto(rep_dcp, rep_dcp_proto);

  PYDEFINE_PROTO(m, SimpleShardedDimProto)
      .PYFIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .PYFIELD_STR(SimpleShardedDimProto, dim_param)
      .PYFIELD(SimpleShardedDimProto, num_shards);
  PYADD_PROTO_SERIALIZATION(SimpleShardedDimProto);
  nb_SimpleShardedDimProto.def(
      "HasField",
      [](const SimpleShardedDimProto &self, const std::string &field_name) -> bool {
        if (field_name == "dim_value")
          return self.has_dim_value();
        if (field_name == "dim_param")
          return self.has_dim_param();
        if (field_name == "num_shards")
          return self.has_num_shards();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(SimpleShardedDimProto, rep_ssdp);
  define_repeated_field_type_proto(rep_ssdp, rep_ssdp_proto);

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding);
  PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  nb_ShardedDimProto.def(
      "HasField",
      [](const ShardedDimProto &self, const std::string &field_name) -> bool {
        if (field_name == "axis")
          return self.has_axis();
        if (field_name == "simple_sharding")
          return self.has_simple_sharding();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(ShardedDimProto, rep_sdp);
  define_repeated_field_type_proto(rep_sdp, rep_sdp_proto);

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim);
  PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  nb_ShardingSpecProto.def(
      "HasField",
      [](const ShardingSpecProto &self, const std::string &field_name) -> bool {
        if (field_name == "tensor_name")
          return self.has_tensor_name();
        if (field_name == "device")
          return self.has_device();
        if (field_name == "index_to_device_group_map")
          return self.has_index_to_device_group_map();
        if (field_name == "sharded_dim")
          return self.has_sharded_dim();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(ShardingSpecProto, rep_ssp);
  define_repeated_field_type_proto(rep_ssp, rep_ssp_proto);

  PYDEFINE_PROTO(m, NodeDeviceConfigurationProto)
      .PYFIELD_STR(NodeDeviceConfigurationProto, configuration_id)
      .PYFIELD(NodeDeviceConfigurationProto, sharding_spec)
      .PYFIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage);
  PYADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);
  nb_NodeDeviceConfigurationProto.def(
      "HasField",
      [](const NodeDeviceConfigurationProto &self, const std::string &field_name) -> bool {
        if (field_name == "configuration_id")
          return self.has_configuration_id();
        if (field_name == "sharding_spec")
          return self.has_sharding_spec();
        if (field_name == "pipeline_stage")
          return self.has_pipeline_stage();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(NodeDeviceConfigurationProto, rep_ndcp);
  define_repeated_field_type_proto(rep_ndcp, rep_ndcp_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorShapeProto);
  PYDEFINE_SUBPROTO(nb_TensorShapeProto, TensorShapeProto, Dimension)
      .PYFIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .PYFIELD_STR(TensorShapeProto::Dimension, dim_param)
      .PYFIELD_STR(TensorShapeProto::Dimension, denotation)
      .def(
          "WhichOneof",
          [](const TensorShapeProto::Dimension &self, const std::string &oneof_name) -> nb::object {
            if (oneof_name != "value")
              throw nb::value_error(
                  ("Protocol message TensorShapeProto.Dimension has no oneof field named '" +
                   oneof_name + "'.")
                      .c_str());
            if (self.has_dim_value())
              return nb::str("dim_value");
            if (self.has_dim_param())
              return nb::str("dim_param");
            return nb::none();
          },
          nb::arg("oneof_name"),
          "Returns the name of the active oneof field, or None if no field is set.")
      .def("__repr__",
           [](TensorShapeProto::Dimension &self) { return proto_repr_with_short_line(self); });
  PYADD_SUBPROTO_SERIALIZATION(TensorShapeProto, Dimension);
  nb_sub_TensorShapeProtoDimension.def(
      "HasField",
      [](const TensorShapeProto::Dimension &self, const std::string &field_name) -> bool {
        if (field_name == "dim_value")
          return self.has_dim_value();
        if (field_name == "dim_param")
          return self.has_dim_param();
        if (field_name == "denotation")
          return self.has_denotation();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_SUBPROTO(TensorShapeProto, Dimension, rep_tspd);
  define_repeated_field_type_proto(rep_tspd, rep_tspd_proto);
  nb_TensorShapeProto.PYFIELD(TensorShapeProto, dim);
  PYADD_PROTO_SERIALIZATION(TensorShapeProto);
  nb_TensorShapeProto.def(
      "HasField",
      [](const TensorShapeProto &self, const std::string &field_name) -> bool {
        if (field_name == "dim")
          return self.has_dim();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  nb_TensorShapeProto.def("__repr__",
                          [](TensorShapeProto &self) { return proto_repr_with_short_line(self); });

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorProto);

  nb::enum_<TensorProto::DataType>(nb_TensorProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", TensorProto::DataType::UNDEFINED)
      .value("FLOAT", TensorProto::DataType::FLOAT)
      .value("UINT8", TensorProto::DataType::UINT8)
      .value("INT8", TensorProto::DataType::INT8)
      .value("UINT16", TensorProto::DataType::UINT16)
      .value("INT16", TensorProto::DataType::INT16)
      .value("INT32", TensorProto::DataType::INT32)
      .value("INT64", TensorProto::DataType::INT64)
      .value("STRING", TensorProto::DataType::STRING)
      .value("BOOL", TensorProto::DataType::BOOL)
      .value("FLOAT16", TensorProto::DataType::FLOAT16)
      .value("DOUBLE", TensorProto::DataType::DOUBLE)
      .value("UINT32", TensorProto::DataType::UINT32)
      .value("UINT64", TensorProto::DataType::UINT64)
      .value("COMPLEX64", TensorProto::DataType::COMPLEX64)
      .value("COMPLEX128", TensorProto::DataType::COMPLEX128)
      .value("BFLOAT16", TensorProto::DataType::BFLOAT16)
      .value("FLOAT8E4M3FN", TensorProto::DataType::FLOAT8E4M3FN)
      .value("FLOAT8E4M3FNUZ", TensorProto::DataType::FLOAT8E4M3FNUZ)
      .value("FLOAT8E5M2", TensorProto::DataType::FLOAT8E5M2)
      .value("FLOAT8E5M2FNUZ", TensorProto::DataType::FLOAT8E5M2FNUZ)
      .value("UINT4", TensorProto::DataType::UINT4)
      .value("INT4", TensorProto::DataType::INT4)
      .value("FLOAT4E2M1", TensorProto::DataType::FLOAT4E2M1)
      .value("FLOAT8E8M0", TensorProto::DataType::FLOAT8E8M0)
      .value("UINT2", TensorProto::DataType::UINT2)
      .value("INT2", TensorProto::DataType::INT2)
      .export_values();
  nb::enum_<TensorProto::DataLocation>(nb_TensorProto, "DataLocation", nb::is_arithmetic())
      .value("DEFAULT", TensorProto::DataLocation::DEFAULT)
      .value("EXTERNAL", TensorProto::DataLocation::EXTERNAL)
      .export_values();
  nb_TensorProto.SHORTEN_CODE(TensorProto::DataType, UNDEFINED)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT)
      .SHORTEN_CODE(TensorProto::DataType, UINT8)
      .SHORTEN_CODE(TensorProto::DataType, INT8)
      .SHORTEN_CODE(TensorProto::DataType, UINT16)
      .SHORTEN_CODE(TensorProto::DataType, INT16)
      .SHORTEN_CODE(TensorProto::DataType, INT32)
      .SHORTEN_CODE(TensorProto::DataType, INT64)
      .SHORTEN_CODE(TensorProto::DataType, STRING)
      .SHORTEN_CODE(TensorProto::DataType, BOOL)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, DOUBLE)
      .SHORTEN_CODE(TensorProto::DataType, UINT32)
      .SHORTEN_CODE(TensorProto::DataType, UINT64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX128)
      .SHORTEN_CODE(TensorProto::DataType, BFLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FN)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, UINT4)
      .SHORTEN_CODE(TensorProto::DataType, INT4)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT4E2M1)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E8M0)
      .SHORTEN_CODE(TensorProto::DataType, UINT2)
      .SHORTEN_CODE(TensorProto::DataType, INT2)
      .PYFIELD(TensorProto, dims)
      .def_prop_rw(
          "data_type",
          [](const TensorProto &self) -> TensorProto::DataType { return self.data_type_; },
          [](TensorProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.data_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.data_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TensorProto::DOC_data_type)
      .def_prop_rw(
          "data_location",
          [](const TensorProto &self) -> TensorProto::DataLocation {
            return self.has_data_location() ? *self.data_location_
                                            : TensorProto::DataLocation::DEFAULT;
          },
          [](TensorProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.data_location_ = static_cast<TensorProto::DataLocation>(nb::cast<int>(obj));
            } else {
              self.data_location_ = nb::cast<TensorProto::DataLocation>(obj);
            }
          },
          TensorProto::DOC_data_location)
      .PYFIELD_STR(TensorProto, name)
      .PYFIELD_STR(TensorProto, doc_string)
      .PYFIELD(TensorProto, external_data)
      .PYFIELD(TensorProto, metadata_props)
      .PYFIELD(TensorProto, dims)
      .PYFIELD(TensorProto, double_data)
      .PYFIELD(TensorProto, float_data)
      .PYFIELD(TensorProto, int64_data)
      .PYFIELD(TensorProto, int32_data)
      .PYFIELD(TensorProto, uint64_data)
      .PYFIELD_REPEATED_STR(TensorProto, string_data)
      .def_prop_rw(
          "raw_data",
          [](const TensorProto &self) -> nb::bytes {
            return nb::bytes(reinterpret_cast<const char *>(self.raw_data_.data()),
                             self.raw_data_.size());
          },
          [](TensorProto &self, nb::bytes data) {
            std::string raw(static_cast<const char *>(data.data()), data.size());
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            self.raw_data_.resize(raw.size());
            memcpy(self.raw_data_.data(), ptr, raw.size());
          },
          TensorProto::DOC_raw_data)
      .def_prop_ro(
          "size",
          [](const TensorProto &self) -> int64_t {
            int64_t size = 1;
            for (auto &it : self.ref_dims()) {
              size *= static_cast<int64_t>(it);
            }
            return size;
          },
          "Returns the number of elements in the tensor.")
      .def(
          "load_external_data",
          [](TensorProto &self, const std::string &base_dir) { self.LoadExternalData(base_dir); },
          nb::arg("base_dir") = std::string(),
          "Loads the raw bytes of this tensor from the external file described by its "
          "``external_data`` field into ``raw_data``. The ``external_data`` and ``data_location`` "
          "fields are preserved.")
      .def(
          "HasField",
          [](const TensorProto &self, const std::string &field_name) -> bool {
            if (field_name == "dims")
              return self.has_dims();
            if (field_name == "data_type")
              return self.has_data_type();
            if (field_name == "segment")
              return self.has_segment();
            if (field_name == "float_data")
              return self.has_float_data();
            if (field_name == "int32_data")
              return self.has_int32_data();
            if (field_name == "string_data")
              return self.has_string_data();
            if (field_name == "int64_data")
              return self.has_int64_data();
            if (field_name == "name")
              return self.has_name();
            if (field_name == "raw_data")
              return self.has_raw_data();
            if (field_name == "double_data")
              return self.has_double_data();
            if (field_name == "uint64_data")
              return self.has_uint64_data();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "external_data")
              return self.has_external_data();
            if (field_name == "data_location")
              return self.has_data_location();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  PYADD_PROTO_SERIALIZATION(TensorProto);
  DECLARE_REPEATED_FIELD_PROTO(TensorProto, rep_tp);
  define_repeated_field_type_proto(rep_tp, rep_tp_proto);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims);
  PYADD_PROTO_SERIALIZATION(SparseTensorProto);
  nb_SparseTensorProto.def(
      "HasField",
      [](const SparseTensorProto &self, const std::string &field_name) -> bool {
        if (field_name == "values")
          return self.has_values();
        if (field_name == "indices")
          return self.has_indices();
        if (field_name == "dims")
          return self.has_dims();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  DECLARE_REPEATED_FIELD_PROTO(SparseTensorProto, rep_tsp);
  define_repeated_field_type_proto(rep_tsp, rep_tsp_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TypeProto);

  PYDEFINE_PROTO_WITH_SUBTYPES2(m, TypeProto, Tensor);
  nb_sub_TypeProtoTensor
      .def_prop_rw(
          "elem_type",
          [](const TypeProto::Tensor &self) -> TensorProto::DataType { return *self.elem_type_; },
          [](TypeProto::Tensor &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TypeProto::Tensor::DOC_elem_type)
      .def("has_elem_type", &TypeProto::Tensor::has_elem_type, "Tells if 'elem_type' has a value.")
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Tensor);
  nb_sub_TypeProtoTensor.def(
      "HasField",
      [](const TypeProto::Tensor &self, const std::string &field_name) -> bool {
        if (field_name == "elem_type")
          return self.has_elem_type();
        if (field_name == "shape")
          return self.has_shape();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");

  PYDEFINE_PROTO_WITH_SUBTYPES2(m, TypeProto, SparseTensor);
  nb_sub_TypeProtoSparseTensor
      .def_prop_rw(
          "elem_type",
          [](const TypeProto::SparseTensor &self) -> TensorProto::DataType {
            return *self.elem_type_;
          },
          [](TypeProto::SparseTensor &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TypeProto::SparseTensor::DOC_elem_type)
      .def("has_elem_type", &TypeProto::SparseTensor::has_elem_type,
           "Tells if 'elem_type' has a value.")
      .PYFIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);
  nb_sub_TypeProtoSparseTensor.def(
      "HasField",
      [](const TypeProto::SparseTensor &self, const std::string &field_name) -> bool {
        if (field_name == "elem_type")
          return self.has_elem_type();
        if (field_name == "shape")
          return self.has_shape();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");

  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Sequence)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Sequence);
  nb_sub_TypeProtoSequence.def(
      "HasField",
      [](const TypeProto::Sequence &self, const std::string &field_name) -> bool {
        if (field_name == "elem_type")
          return self.has_elem_type();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Optional)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Optional);
  nb_sub_TypeProtoOptional.def(
      "HasField",
      [](const TypeProto::Optional &self, const std::string &field_name) -> bool {
        if (field_name == "elem_type")
          return self.has_elem_type();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Map)
      .PYFIELD(TypeProto::Map, key_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Map, value_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Map);
  nb_sub_TypeProtoMap.def(
      "HasField",
      [](const TypeProto::Map &self, const std::string &field_name) -> bool {
        if (field_name == "key_type")
          return self.has_key_type();
        if (field_name == "value_type")
          return self.has_value_type();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");
  nb_TypeProto.PYFIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .PYFIELD_STR(TypeProto, denotation)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, optional_type)
      .def(
          "WhichOneof",
          [](const TypeProto &self, const std::string &oneof_name) -> nb::object {
            if (oneof_name != "value")
              throw nb::value_error(
                  ("Protocol message TypeProto has no oneof field named '" + oneof_name + "'.")
                      .c_str());
            if (self.has_tensor_type())
              return nb::str("tensor_type");
            if (self.has_sequence_type())
              return nb::str("sequence_type");
            if (self.has_map_type())
              return nb::str("map_type");
            if (self.has_sparse_tensor_type())
              return nb::str("sparse_tensor_type");
            if (self.has_optional_type())
              return nb::str("optional_type");
            return nb::none();
          },
          nb::arg("oneof_name"),
          "Returns the name of the field set in the oneof ``oneof_name``, or None if no field is "
          "set, following the protobuf API.");
  PYADD_PROTO_SERIALIZATION(TypeProto);
  nb_TypeProto
      .def(
          "HasField",
          [](const TypeProto &self, const std::string &field_name) -> bool {
            if (field_name == "tensor_type")
              return self.has_tensor_type();
            if (field_name == "sequence_type")
              return self.has_sequence_type();
            if (field_name == "map_type")
              return self.has_map_type();
            if (field_name == "sparse_tensor_type")
              return self.has_sparse_tensor_type();
            if (field_name == "optional_type")
              return self.has_optional_type();
            if (field_name == "denotation")
              return self.has_denotation();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def("__repr__", [](TypeProto &self) { return proto_repr_with_short_line(self); });
  DECLARE_REPEATED_FIELD_PROTO(TypeProto, rep_typeproto);
  define_repeated_field_type_proto(rep_typeproto, rep_typeproto_proto);

  PYDEFINE_PROTO(m, ValueInfoProto)
      .PYFIELD_STR(ValueInfoProto, name)
      .PYFIELD_OPTIONAL_PROTO(ValueInfoProto, type)
      .PYFIELD_STR(ValueInfoProto, doc_string)
      .PYFIELD(ValueInfoProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(ValueInfoProto);
  nb_ValueInfoProto
      .def(
          "HasField",
          [](const ValueInfoProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "type")
              return self.has_type();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def("__repr__", [](ValueInfoProto &self) { return proto_repr_with_short_line(self); });
  DECLARE_REPEATED_FIELD_PROTO(ValueInfoProto, rep_vip);
  define_repeated_field_type_proto(rep_vip, rep_vip_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, AttributeProto);
  nb::enum_<AttributeProto::AttributeType> attribute_type(nb_AttributeProto, "AttributeType",
                                                          nb::is_arithmetic());
  attribute_type.value("UNDEFINED", AttributeProto::AttributeType::UNDEFINED)
      .value("FLOAT", AttributeProto::AttributeType::FLOAT)
      .value("INT", AttributeProto::AttributeType::INT)
      .value("STRING", AttributeProto::AttributeType::STRING)
      .value("TENSOR", AttributeProto::AttributeType::TENSOR)
      .value("GRAPH", AttributeProto::AttributeType::GRAPH)
      .value("SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR)
      .value("FLOATS", AttributeProto::AttributeType::FLOATS)
      .value("INTS", AttributeProto::AttributeType::INTS)
      .value("STRINGS", AttributeProto::AttributeType::STRINGS)
      .value("TENSORS", AttributeProto::AttributeType::TENSORS)
      .value("GRAPHS", AttributeProto::AttributeType::GRAPHS)
      .value("SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS)
      .value("TYPE_PROTO", AttributeProto::AttributeType::TYPE_PROTO)
      .value("TYPE_PROTOS", AttributeProto::AttributeType::TYPE_PROTOS)
      .export_values();
  attribute_type
      .def_static(
          "items",
          []() {
            return std::vector<std::pair<std::string, AttributeProto::AttributeType>>{
                {"UNDEFINED", AttributeProto::AttributeType::UNDEFINED},
                {"FLOAT", AttributeProto::AttributeType::FLOAT},
                {"INT", AttributeProto::AttributeType::INT},
                {"STRING", AttributeProto::AttributeType::STRING},
                {"TENSOR", AttributeProto::AttributeType::TENSOR},
                {"GRAPH", AttributeProto::AttributeType::GRAPH},
                {"SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR},
                {"FLOATS", AttributeProto::AttributeType::FLOATS},
                {"INTS", AttributeProto::AttributeType::INTS},
                {"STRINGS", AttributeProto::AttributeType::STRINGS},
                {"TENSORS", AttributeProto::AttributeType::TENSORS},
                {"GRAPHS", AttributeProto::AttributeType::GRAPHS},
                {"SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS},
            };
          },
          "Returns the list of (name, type).")
      .def_static(
          "keys",
          []() {
            return std::vector<std::string>{
                "UNDEFINED",      "FLOAT",  "INT",  "STRING",  "TENSOR",  "GRAPH",
                "SPARSE_TENSOR",  "FLOATS", "INTS", "STRINGS", "TENSORS", "GRAPHS",
                "SPARSE_TENSORS",
            };
          },
          "Returns the list of names.")
      .def_static(
          "values",
          []() {
            return std::vector<AttributeProto::AttributeType>{
                AttributeProto::AttributeType::UNDEFINED,
                AttributeProto::AttributeType::FLOAT,
                AttributeProto::AttributeType::INT,
                AttributeProto::AttributeType::STRING,
                AttributeProto::AttributeType::TENSOR,
                AttributeProto::AttributeType::GRAPH,
                AttributeProto::AttributeType::SPARSE_TENSOR,
                AttributeProto::AttributeType::FLOATS,
                AttributeProto::AttributeType::INTS,
                AttributeProto::AttributeType::STRINGS,
                AttributeProto::AttributeType::TENSORS,
                AttributeProto::AttributeType::GRAPHS,
                AttributeProto::AttributeType::SPARSE_TENSORS,
            };
          },
          "Returns the list of types.");

  nb_AttributeProto.SHORTEN_CODE(AttributeProto::AttributeType, UNDEFINED)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOAT)
      .SHORTEN_CODE(AttributeProto::AttributeType, INT)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRING)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPH)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOATS)
      .SHORTEN_CODE(AttributeProto::AttributeType, INTS)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRINGS)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSORS)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPHS)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSORS)
      .PYFIELD_STR(AttributeProto, name)
      .PYFIELD_STR(AttributeProto, ref_attr_name)
      .PYFIELD_STR(AttributeProto, doc_string)
      .def_prop_rw(
          "type",
          [](const AttributeProto &self) -> AttributeProto::AttributeType { return self.type_; },
          [](AttributeProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.type_ = static_cast<AttributeProto::AttributeType>(nb::cast<int>(obj));
            } else {
              self.type_ = nb::cast<AttributeProto::AttributeType>(obj);
            }
          },
          AttributeProto::DOC_type)
      .PYFIELD_OPTIONAL_FLOAT(AttributeProto, f)
      .PYFIELD_OPTIONAL_INT(AttributeProto, i)
      .PYFIELD_STR_AS_BYTES(AttributeProto, s)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, t)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, sparse_tensor)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, g)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, tp)
      .PYFIELD(AttributeProto, floats)
      .PYFIELD(AttributeProto, ints)
      .PYFIELD_REPEATED_STR(AttributeProto, strings)
      .PYFIELD(AttributeProto, tensors)
      .PYFIELD(AttributeProto, sparse_tensors)
      .PYFIELD(AttributeProto, graphs)
      .PYFIELD(AttributeProto, type_protos);
  PYADD_PROTO_SERIALIZATION(AttributeProto);
  nb_AttributeProto
      .def(
          "HasField",
          [](const AttributeProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "ref_attr_name")
              return self.has_ref_attr_name();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "f")
              return self.has_f();
            if (field_name == "i")
              return self.has_i();
            if (field_name == "s")
              return self.has_s();
            if (field_name == "t")
              return self.has_t();
            if (field_name == "sparse_tensor")
              return self.has_sparse_tensor();
            if (field_name == "g")
              return self.has_g();
            if (field_name == "tp")
              return self.has_tp();
            if (field_name == "floats")
              return self.has_floats();
            if (field_name == "ints")
              return self.has_ints();
            if (field_name == "strings")
              return self.has_strings();
            if (field_name == "tensors")
              return self.has_tensors();
            if (field_name == "sparse_tensors")
              return self.has_sparse_tensors();
            if (field_name == "graphs")
              return self.has_graphs();
            if (field_name == "type_protos")
              return self.has_type_protos();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def("__repr__", [](AttributeProto &self) { return proto_repr_with_short_line(self); });
  DECLARE_REPEATED_FIELD_PROTO(AttributeProto, rep_ap);
  define_repeated_field_type_proto(rep_ap, rep_ap_proto);

  PYDEFINE_PROTO(m, NodeProto)
      .PYFIELD(NodeProto, input)
      .PYFIELD(NodeProto, output)
      .PYFIELD_STR(NodeProto, name)
      .PYFIELD_STR(NodeProto, op_type)
      .PYFIELD_STR(NodeProto, domain)
      .PYFIELD_STR(NodeProto, overload)
      .PYFIELD(NodeProto, attribute)
      .PYFIELD_STR(NodeProto, doc_string)
      .PYFIELD(NodeProto, metadata_props)
      .PYFIELD(NodeProto, device_configurations);
  PYADD_PROTO_SERIALIZATION(NodeProto);
  nb_NodeProto
      .def(
          "HasField",
          [](const NodeProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "op_type")
              return self.has_op_type();
            if (field_name == "domain")
              return self.has_domain();
            if (field_name == "overload")
              return self.has_overload();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "input")
              return self.has_input();
            if (field_name == "output")
              return self.has_output();
            if (field_name == "attribute")
              return self.has_attribute();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            if (field_name == "device_configurations")
              return self.has_device_configurations();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def("__repr__", [](NodeProto &self) { return proto_repr_with_short_line(self); });
  nb_NodeProto
      .def(
          "set_attribute",
          [](NodeProto &self, const std::string &name, nb::object value, nb::object attr_type,
             nb::object doc_string) -> AttributeProto & {
            nb::module_ helper = ImportHelper();
            nb::object built = helper.attr("make_attribute")(
                name, value, nb::arg("doc_string") = doc_string, nb::arg("attr_type") = attr_type);
            return self.set_attribute(nb::cast<const AttributeProto &>(built));
          },
          nb::arg("name"), nb::arg("value"), nb::arg("attr_type") = nb::none(),
          nb::arg("doc_string") = nb::none(), nb::rv_policy::reference_internal,
          "Sets attribute *name* on this node to *value*, replacing an existing attribute with "
          "the same name in place. The attribute type is inferred from *value* via "
          "``onnx_light.onnx.helper.make_attribute``; pass *attr_type* to disambiguate.")
      .def("add_metadata", &NodeProto::add_metadata, nb::arg("key"), nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets metadata property *key* to *value*, updating an existing entry in place.");
  DECLARE_REPEATED_FIELD_PROTO(NodeProto, rep_node);
  define_repeated_field_type_proto(rep_node, rep_node_proto);

  PYDEFINE_PROTO(m, GraphProto)
      .PYFIELD(GraphProto, node)
      .PYFIELD_STR(GraphProto, name)
      .PYFIELD(GraphProto, initializer)
      .PYFIELD(GraphProto, sparse_initializer)
      .PYFIELD_STR(GraphProto, doc_string)
      .PYFIELD(GraphProto, input)
      .PYFIELD(GraphProto, output)
      .PYFIELD(GraphProto, value_info)
      .PYFIELD(GraphProto, quantization_annotation)
      .PYFIELD(GraphProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(GraphProto);
  nb_GraphProto
      .def(
          "HasField",
          [](const GraphProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "node")
              return self.has_node();
            if (field_name == "initializer")
              return self.has_initializer();
            if (field_name == "sparse_initializer")
              return self.has_sparse_initializer();
            if (field_name == "input")
              return self.has_input();
            if (field_name == "output")
              return self.has_output();
            if (field_name == "value_info")
              return self.has_value_info();
            if (field_name == "quantization_annotation")
              return self.has_quantization_annotation();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def(
          "add_input",
          [](GraphProto &self, nb::object name_or_proto, nb::object elem_type, nb::object shape,
             nb::object doc_string) -> ValueInfoProto & {
            self.input_.push_back(MakeValueInfoForPy(name_or_proto, elem_type, shape, doc_string));
            return self.input_.back();
          },
          nb::arg("name_or_proto"), nb::arg("elem_type") = nb::none(),
          nb::arg("shape") = nb::none(), nb::arg("doc_string") = nb::none(),
          nb::rv_policy::reference_internal,
          "Appends a new input and returns it. Accepts either a prebuilt ``ValueInfoProto`` or "
          "``(name, elem_type, shape)``.")
      .def(
          "add_output",
          [](GraphProto &self, nb::object name_or_proto, nb::object elem_type, nb::object shape,
             nb::object doc_string) -> ValueInfoProto & {
            self.output_.push_back(MakeValueInfoForPy(name_or_proto, elem_type, shape, doc_string));
            return self.output_.back();
          },
          nb::arg("name_or_proto"), nb::arg("elem_type") = nb::none(),
          nb::arg("shape") = nb::none(), nb::arg("doc_string") = nb::none(),
          nb::rv_policy::reference_internal,
          "Appends a new output and returns it. See :meth:`add_input`.")
      .def(
          "add_value_info",
          [](GraphProto &self, nb::object name_or_proto, nb::object elem_type, nb::object shape,
             nb::object doc_string) -> ValueInfoProto & {
            self.value_info_.push_back(
                MakeValueInfoForPy(name_or_proto, elem_type, shape, doc_string));
            return self.value_info_.back();
          },
          nb::arg("name_or_proto"), nb::arg("elem_type") = nb::none(),
          nb::arg("shape") = nb::none(), nb::arg("doc_string") = nb::none(),
          nb::rv_policy::reference_internal,
          "Appends a new intermediate value_info entry and returns it.")
      .def(
          "add_initializer",
          [](GraphProto &self, nb::object name_or_proto, nb::object array) -> TensorProto & {
            if (nb::isinstance<TensorProto>(name_or_proto)) {
              if (!array.is_none()) {
                throw nb::value_error("array must be None when a TensorProto is passed.");
              }
              self.initializer_.push_back(nb::cast<TensorProto>(name_or_proto));
            } else {
              if (array.is_none()) {
                throw nb::value_error("array is required when a name is passed.");
              }
              nb::module_ numpy_helper = ImportNumpyHelper();
              nb::object built =
                  numpy_helper.attr("from_array")(array, nb::arg("name") = name_or_proto);
              self.initializer_.push_back(nb::cast<TensorProto>(built));
            }
            return self.initializer_.back();
          },
          nb::arg("name_or_proto"), nb::arg("array") = nb::none(),
          nb::rv_policy::reference_internal,
          "Appends a new initializer and returns it. Accepts either a prebuilt ``TensorProto`` "
          "or ``(name, numpy_array)``.")
      .def(
          "add_node",
          [](GraphProto &self, nb::object op_type, nb::object inputs, nb::object outputs,
             nb::kwargs kwargs) -> NodeProto & {
            return AddNodeImpl(self, op_type, inputs, outputs, kwargs);
          },
          nb::rv_policy::reference_internal,
          "Builds a :class:`NodeProto` via ``onnx_light.onnx.helper.make_node`` (extra keyword "
          "arguments become node attributes) and appends it.")
      .def("add_metadata", &GraphProto::add_metadata, nb::arg("key"), nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets metadata property *key* to *value*, updating an existing entry in place.");
  DECLARE_REPEATED_FIELD_PROTO(GraphProto, rep_graph);
  define_repeated_field_type_proto(rep_graph, rep_graph_proto);

  PYDEFINE_PROTO(m, FunctionProto)
      .PYFIELD_STR(FunctionProto, name)
      .PYFIELD(FunctionProto, input)
      .PYFIELD(FunctionProto, output)
      .PYFIELD(FunctionProto, attribute)
      .PYFIELD(FunctionProto, attribute_proto)
      .PYFIELD(FunctionProto, node)
      .PYFIELD_STR(FunctionProto, doc_string)
      .PYFIELD(FunctionProto, opset_import)
      .PYFIELD_STR(FunctionProto, domain)
      .PYFIELD_STR(FunctionProto, overload)
      .PYFIELD(FunctionProto, value_info)
      .PYFIELD(FunctionProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(FunctionProto);
  nb_FunctionProto
      .def(
          "HasField",
          [](const FunctionProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "domain")
              return self.has_domain();
            if (field_name == "overload")
              return self.has_overload();
            if (field_name == "input")
              return self.has_input();
            if (field_name == "output")
              return self.has_output();
            if (field_name == "attribute")
              return self.has_attribute();
            if (field_name == "attribute_proto")
              return self.has_attribute_proto();
            if (field_name == "node")
              return self.has_node();
            if (field_name == "opset_import")
              return self.has_opset_import();
            if (field_name == "value_info")
              return self.has_value_info();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def(
          "add_input",
          [](FunctionProto &self, const std::string &name) {
            self.input_.push_back(utils::String(name));
          },
          nb::arg("name"), "Appends an input name to the function.")
      .def(
          "add_output",
          [](FunctionProto &self, const std::string &name) {
            self.output_.push_back(utils::String(name));
          },
          nb::arg("name"), "Appends an output name to the function.")
      .def(
          "add_node",
          [](FunctionProto &self, nb::object op_type, nb::object inputs, nb::object outputs,
             nb::kwargs kwargs) -> NodeProto & {
            return AddNodeImpl(self, op_type, inputs, outputs, kwargs);
          },
          nb::rv_policy::reference_internal,
          "Builds a :class:`NodeProto` via ``onnx_light.onnx.helper.make_node`` (extra keyword "
          "arguments become node attributes) and appends it.")
      .def("add_opset", &FunctionProto::add_opset, nb::arg("domain"), nb::arg("version"),
           nb::rv_policy::reference_internal, "Appends an opset import ``(domain, version)``.")
      .def("add_metadata", &FunctionProto::add_metadata, nb::arg("key"), nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets metadata property *key* to *value*, updating an existing entry in place.");
  DECLARE_REPEATED_FIELD_PROTO(FunctionProto, rep_function);
  define_repeated_field_type_proto(rep_function, rep_function_proto);

  PYDEFINE_PROTO(m, ModelProto)
      .PYFIELD_STR(ModelProto, producer_name)
      .PYFIELD_STR(ModelProto, producer_version)
      .PYFIELD_STR(ModelProto, domain)
      .PYFIELD_OPTIONAL_INT(ModelProto, model_version)
      .PYFIELD_STR(ModelProto, doc_string)
      .PYFIELD_OPTIONAL_PROTO(ModelProto, graph)
      .PYFIELD(ModelProto, opset_import)
      .PYFIELD_OPTIONAL_INT(ModelProto, ir_version)
      .PYFIELD(ModelProto, metadata_props)
      .PYFIELD(ModelProto, functions)
      .PYFIELD(ModelProto, configuration);
  PYADD_PROTO_SERIALIZATION(ModelProto);
  nb_ModelProto
      .def(
          "HasField",
          [](const ModelProto &self, const std::string &field_name) -> bool {
            if (field_name == "ir_version")
              return self.has_ir_version();
            if (field_name == "producer_name")
              return self.has_producer_name();
            if (field_name == "producer_version")
              return self.has_producer_version();
            if (field_name == "domain")
              return self.has_domain();
            if (field_name == "model_version")
              return self.has_model_version();
            if (field_name == "doc_string")
              return self.has_doc_string();
            if (field_name == "graph")
              return self.has_graph();
            if (field_name == "metadata_props")
              return self.has_metadata_props();
            if (field_name == "opset_import")
              return self.has_opset_import();
            if (field_name == "functions")
              return self.has_functions();
            if (field_name == "configuration")
              return self.has_configuration();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def("__repr__", [](ModelProto &self) { return proto_repr_with_short_line(self); });
  nb_ModelProto
      .def("add_function", &ModelProto::add_function, nb::arg("function"),
           nb::rv_policy::reference_internal, "Appends a :class:`FunctionProto` and returns it.")
      .def("add_opset", &ModelProto::add_opset, nb::arg("domain"), nb::arg("version"),
           nb::rv_policy::reference_internal, "Appends an opset import ``(domain, version)``.")
      .def("add_metadata", &ModelProto::add_metadata, nb::arg("key"), nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets metadata property *key* to *value*, updating an existing entry in place.");
#ifdef ONNX_LIGHT_HAS_OPENSSL
  nb_ModelProto
      .def(
          "SerializeToEncryptedFile",
          [](ModelProto &self, const std::string &file_path, const std::string &key,
             nb::object options, const std::string &encryption) {
            SerializeOptions opts;
            if (nb::isinstance<SerializeOptions>(options)) {
              opts = nb::cast<SerializeOptions>(options);
            }
            SaveEncryptedModel(self, file_path, key, opts, encryption);
          },
          nb::arg("name"), nb::arg("key"), nb::arg("options") = nb::none(),
          nb::arg("encryption") = "AES-256-CBC",
          "Encrypts the model and writes it to a single binary file. Supported values are "
          "\"AES-256-CBC\" (ONNXCRY1) and \"ChaCha20-Poly1305\" (ONNXCRY2).")
      .def(
          "ParseFromEncryptedFile",
          [](ModelProto &self, const std::string &file_path, const std::string &key,
             nb::object options) {
            ParseOptions opts;
            if (nb::isinstance<ParseOptions>(options)) {
              opts = nb::cast<ParseOptions>(options);
            }
            LoadEncryptedModel(self, file_path, key, opts);
          },
          nb::arg("name"), nb::arg("key"), nb::arg("options") = nb::none(),
          "Decrypts an ONNXCRY1/ONNXCRY2 encrypted file (written by SerializeToEncryptedFile) and "
          "parses the payload into this model instance.")
      .def(
          "SerializeToEncryptedString",
          [](ModelProto &self, const std::string &key, nb::object options,
             const std::string &encryption) {
            SerializeOptions opts;
            if (nb::isinstance<SerializeOptions>(options)) {
              opts = nb::cast<SerializeOptions>(options);
            }
            const std::string blob = SaveEncryptedModelToString(self, key, opts, encryption);
            return nb::bytes(blob.data(), blob.size());
          },
          nb::arg("key"), nb::arg("options") = nb::none(), nb::arg("encryption") = "AES-256-CBC",
          "Encrypts the model and returns ciphertext bytes in ONNXCRY1 (AES-256-CBC) or "
          "ONNXCRY2 (ChaCha20-Poly1305) format.")
      .def(
          "ParseFromEncryptedString",
          [](ModelProto &self, nb::bytes data, const std::string &key, nb::object options) {
            ParseOptions opts;
            if (nb::isinstance<ParseOptions>(options)) {
              opts = nb::cast<ParseOptions>(options);
            }
            const std::string blob(reinterpret_cast<const char *>(data.data()), data.size());
            LoadEncryptedModelFromString(self, blob, key, opts);
          },
          nb::arg("data"), nb::arg("key"), nb::arg("options") = nb::none(),
          "Decrypts an ONNXCRY1/ONNXCRY2 encrypted bytes object (produced by "
          "SerializeToEncryptedString) "
          "and parses the payload into this model instance.");
#endif // ONNX_LIGHT_HAS_OPENSSL

  PYDEFINE_PROTO_WITH_SUBTYPES(m, SequenceProto);
  nb::enum_<SequenceProto::DataType>(nb_SequenceProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", SequenceProto::DataType::UNDEFINED)
      .value("TENSOR", SequenceProto::DataType::TENSOR)
      .value("SPARSE_TENSOR", SequenceProto::DataType::SPARSE_TENSOR)
      .value("SEQUENCE", SequenceProto::DataType::SEQUENCE)
      .value("MAP", SequenceProto::DataType::MAP)
      .value("OPTIONAL", SequenceProto::DataType::OPTIONAL)
      .export_values();
  nb_SequenceProto.SHORTEN_CODE(SequenceProto::DataType, UNDEFINED)
      .SHORTEN_CODE(SequenceProto::DataType, TENSOR)
      .SHORTEN_CODE(SequenceProto::DataType, SPARSE_TENSOR)
      .SHORTEN_CODE(SequenceProto::DataType, SEQUENCE)
      .SHORTEN_CODE(SequenceProto::DataType, MAP)
      .SHORTEN_CODE(SequenceProto::DataType, OPTIONAL);
  nb_SequenceProto.PYFIELD_STR(SequenceProto, name)
      .def_prop_rw(
          "elem_type",
          [](const SequenceProto &self) -> SequenceProto::DataType { return self.elem_type_; },
          [](SequenceProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<SequenceProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<SequenceProto::DataType>(obj);
            }
          },
          SequenceProto::DOC_elem_type)
      .PYFIELD(SequenceProto, tensor_values)
      .PYFIELD(SequenceProto, sparse_tensor_values)
      .PYFIELD(SequenceProto, sequence_values)
      .PYFIELD(SequenceProto, map_values)
      .PYFIELD(SequenceProto, optional_values);
  PYADD_PROTO_SERIALIZATION(SequenceProto);
  nb_SequenceProto.def(
      "HasField",
      [](const SequenceProto &self, const std::string &field_name) -> bool {
        if (field_name == "name")
          return self.has_name();
        if (field_name == "elem_type")
          return self.has_elem_type();
        if (field_name == "tensor_values")
          return self.has_tensor_values();
        if (field_name == "sparse_tensor_values")
          return self.has_sparse_tensor_values();
        if (field_name == "sequence_values")
          return self.has_sequence_values();
        if (field_name == "map_values")
          return self.has_map_values();
        if (field_name == "optional_values")
          return self.has_optional_values();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");

  PYDEFINE_PROTO_WITH_SUBTYPES(m, MapProto);
  nb_MapProto.PYFIELD_STR(MapProto, name)
      .def_prop_rw(
          "key_type", [](const MapProto &self) -> TensorProto::DataType { return self.key_type_; },
          [](MapProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.key_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.key_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          MapProto::DOC_key_type)
      .PYFIELD(MapProto, keys)
      .PYFIELD(MapProto, string_keys)
      .PYFIELD(MapProto, values);
  PYADD_PROTO_SERIALIZATION(MapProto);
  nb_MapProto.def(
      "HasField",
      [](const MapProto &self, const std::string &field_name) -> bool {
        if (field_name == "name")
          return self.has_name();
        if (field_name == "key_type")
          return self.has_key_type();
        if (field_name == "keys")
          return self.has_keys();
        if (field_name == "string_keys")
          return self.has_string_keys();
        if (field_name == "values")
          return self.has_values();
        throw nb::attribute_error(
            ("Protocol message has no field named '" + field_name + "'").c_str());
      },
      nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.");

  PYDEFINE_PROTO_WITH_SUBTYPES(m, OptionalProto);
  nb::enum_<OptionalProto::DataType>(nb_OptionalProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", OptionalProto::DataType::UNDEFINED)
      .value("TENSOR", OptionalProto::DataType::TENSOR)
      .value("SPARSE_TENSOR", OptionalProto::DataType::SPARSE_TENSOR)
      .value("SEQUENCE", OptionalProto::DataType::SEQUENCE)
      .value("MAP", OptionalProto::DataType::MAP)
      .value("OPTIONAL", OptionalProto::DataType::OPTIONAL)
      .export_values();
  nb_OptionalProto.SHORTEN_CODE(OptionalProto::DataType, UNDEFINED)
      .SHORTEN_CODE(OptionalProto::DataType, TENSOR)
      .SHORTEN_CODE(OptionalProto::DataType, SPARSE_TENSOR)
      .SHORTEN_CODE(OptionalProto::DataType, SEQUENCE)
      .SHORTEN_CODE(OptionalProto::DataType, MAP)
      .SHORTEN_CODE(OptionalProto::DataType, OPTIONAL);
  nb_OptionalProto.PYFIELD_STR(OptionalProto, name)
      .def_prop_rw(
          "elem_type",
          [](const OptionalProto &self) -> OptionalProto::DataType { return self.elem_type_; },
          [](OptionalProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<OptionalProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<OptionalProto::DataType>(obj);
            }
          },
          OptionalProto::DOC_elem_type)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, tensor_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, sparse_tensor_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, sequence_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, map_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, optional_value)
      .def(
          "HasField",
          [](const OptionalProto &self, const std::string &field_name) -> bool {
            if (field_name == "name")
              return self.has_name();
            if (field_name == "elem_type")
              return self.has_elem_type();
            if (field_name == "tensor_value")
              return self.has_tensor_value();
            if (field_name == "sparse_tensor_value")
              return self.has_sparse_tensor_value();
            if (field_name == "sequence_value")
              return self.has_sequence_value();
            if (field_name == "map_value")
              return self.has_map_value();
            if (field_name == "optional_value")
              return self.has_optional_value();
            throw nb::attribute_error(
                ("Protocol message has no field named '" + field_name + "'").c_str());
          },
          nb::arg("field_name"), "Checks if a field is set, following the protobuf HasField API.")
      .def(
          "WhichOneof",
          [](const OptionalProto &self, const std::string &oneof_name) -> nb::object {
            if (oneof_name != "value")
              throw nb::value_error(
                  ("Protocol message OptionalProto has no oneof field named '" + oneof_name + "'.")
                      .c_str());
            if (self.has_tensor_value())
              return nb::str("tensor_value");
            if (self.has_sparse_tensor_value())
              return nb::str("sparse_tensor_value");
            if (self.has_sequence_value())
              return nb::str("sequence_value");
            if (self.has_map_value())
              return nb::str("map_value");
            if (self.has_optional_value())
              return nb::str("optional_value");
            return nb::none();
          },
          nb::arg("oneof_name"),
          "Returns the name of the field set in the oneof ``oneof_name``, or None if no field is "
          "set, following the protobuf API.");
  PYADD_PROTO_SERIALIZATION(OptionalProto);

  // -----------------------------------------------------------------------
  // Submodule `verify`: schema-free structural validation (onnx_verify.h).
  // -----------------------------------------------------------------------
  auto verify_mod = m.def_submodule(
      "verify", "Schema-free structural validation of onnx_proto messages (onnx_verify.h). "
                "Unlike onnx_light.checker.check_model, these functions do not require an "
                "operator-schema registry: they only validate internal consistency of the "
                "protobuf structure itself (required fields set, unique names, SSA form, "
                "topological node ordering, tensor payload/dtype consistency, ...). They raise "
                "ValueError on the first violation found.");

  verify_mod.def(
      "verify_value_info",
      [](const ValueInfoProto &value_info, bool is_main_graph) {
        VerifyValueInfo(value_info, is_main_graph);
      },
      nb::arg("value_info"), nb::arg("is_main_graph") = true,
      "Validates a ValueInfoProto. ``is_main_graph=False`` relaxes the 'type' field "
      "requirement, as allowed for subgraph inputs/outputs. Raises ValueError on failure.");

  verify_mod.def(
      "verify_tensor", [](const TensorProto &tensor) { VerifyTensor(tensor); }, nb::arg("tensor"),
      "Validates a TensorProto: data_type is set and the populated payload field matches "
      "the declared data_type and shape. Raises ValueError on failure.");

  verify_mod.def(
      "verify_sparse_tensor",
      [](const SparseTensorProto &sparse_tensor) { VerifySparseTensor(sparse_tensor); },
      nb::arg("sparse_tensor"),
      "Validates a SparseTensorProto: 'values' and 'indices' are individually valid tensors "
      "and 'indices' uses the required INT64 data type. Raises ValueError on failure.");

  verify_mod.def(
      "verify_attribute",
      [](const AttributeProto &attribute, bool in_function_body,
         const std::unordered_set<std::string> &scope) {
        VerifyAttribute(attribute, in_function_body, scope);
      },
      nb::arg("attribute"), nb::arg("in_function_body") = false,
      nb::arg("scope") = std::unordered_set<std::string>{},
      "Validates an AttributeProto: exactly one value field is set and matches the "
      "declared 'type'. ``scope`` lists names visible to a nested subgraph attribute "
      "(GRAPH/GRAPHS), if any. Raises ValueError on failure.");

  verify_mod.def(
      "verify_node",
      [](const NodeProto &node, bool in_function_body,
         const std::unordered_set<std::string> &scope) {
        VerifyNode(node, in_function_body, scope);
      },
      nb::arg("node"), nb::arg("in_function_body") = false,
      nb::arg("scope") = std::unordered_set<std::string>{},
      "Validates a NodeProto: 'op_type' is set, the node has at least one input or "
      "output, and attribute names are unique and individually valid. Raises ValueError "
      "on failure.");

  verify_mod.def(
      "verify_graph",
      [](const GraphProto &graph, bool is_main_graph, bool in_function_body,
         std::optional<std::unordered_set<std::string>> outer_scope) {
        VerifyGraph(graph, is_main_graph, in_function_body,
                    outer_scope.has_value() ? &*outer_scope : nullptr);
      },
      nb::arg("graph"), nb::arg("is_main_graph") = true, nb::arg("in_function_body") = false,
      nb::arg("outer_scope") = nb::none(),
      "Validates a GraphProto: unique input/initializer/output names, SSA form, "
      "topologically sorted nodes, and that every declared output is produced. "
      "``outer_scope`` lists names visible from an enclosing graph, used when validating "
      "a control-flow body subgraph. Raises ValueError on failure.");

  verify_mod.def(
      "verify_function", [](const FunctionProto &function) { VerifyFunction(function); },
      nb::arg("function"),
      "Validates a FunctionProto: 'name' is set, inputs are uniquely named, nodes are "
      "topologically sorted, and every declared output is produced. Raises ValueError on "
      "failure.");

  verify_mod.def(
      "verify_model", [](const ModelProto &model) { VerifyModel(model); }, nb::arg("model"),
      "Validates an in-memory ModelProto without an operator-schema registry: 'graph' is "
      "present, at least one opset is imported with unique domains, the main graph is "
      "structurally valid, and model-local functions are individually valid and uniquely "
      "identified by (domain, name, overload). This only checks IR-level structure; it "
      "does not check operator input/output arity or run shape inference (use "
      "onnx_light.checker.check_model for that). Raises ValueError on failure.");

  // Registers every repeated field container as a virtual subclass of
  // collections.abc.Sequence so that ``isinstance(repeated_field, Sequence)``
  // returns True, matching the behaviour of protobuf repeated containers.
  nb::object sequence_abc = nb::module_::import_("collections.abc").attr("Sequence");
  nb::dict module_dict = nb::borrow<nb::dict>(m.attr("__dict__"));
  for (auto item : module_dict) {
    std::string name = nb::cast<std::string>(item.first);
    if (name.rfind("RepeatedField", 0) == 0 || name.rfind("RepeatedProtoField", 0) == 0)
      sequence_abc.attr("register")(item.second);
  }
}
