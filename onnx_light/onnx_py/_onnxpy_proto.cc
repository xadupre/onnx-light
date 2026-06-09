#include "_onnxpyprotoop.h"
#include "onnx.h"
#include "onnx_crypt.h"
#include "onnx_helper.h"
#include "onnx_lib/onnx-data.pb.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <type_traits>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {
constexpr size_t MAX_SHORT_REPR_LENGTH = 60;
inline bool is_space_char(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

ModelProto MakeOwnedModelProtoCopy(const ModelProto &model) {
  // Fully reparse through bytes to ensure every borrowed span in the model
  // becomes owned before serialization paths that may mutate buffers/metadata.
  std::string serialized;
  model.SerializeToString(serialized);
  ModelProto owned;
  owned.ParseFromString(serialized);
  return owned;
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
      #name,                                                                                       \
      [](const cls &self) -> std::string {                                                         \
        std::string s = self.ref_##name().as_string();                                             \
        return s;                                                                                  \
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

#define PYFIELD_STR_AS_BYTES(cls, name)                                                            \
  def_prop_rw(                                                                                     \
      #name,                                                                                       \
      [](const cls &self) -> nb::bytes {                                                           \
        std::string s = self.ref_##name().as_string();                                             \
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
      cls::DOC_##name)                                                                             \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_INT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, int)
#define PYFIELD_OPTIONAL_FLOAT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, float)

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                          \
  def_prop_rw(                                                                                     \
      #name, [](cls & self) -> cls::name##_t * {                                                   \
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
      nb::rv_policy::reference_internal, cls::DOC_##name)                                          \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")                    \
      .def(                                                                                        \
          "add_" #name, [](cls & self) -> cls::name##_t & {                                        \
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
  name_inst.def(
               "Clear", [](cls &self) { self.CopyFrom(cls()); }, "Clears the object.")
      .def(
          "ParseFromString",
          [](cls &self, nb::bytes data, nb::object options) {
            const uint8_t *bytes_ptr = reinterpret_cast<const uint8_t *>(data.data());
            ONNX_LIGHT_NAMESPACE::utils::StringStream stream(bytes_ptr,
                                                             static_cast<int64_t>(data.size()));
            if (nb::isinstance<ParseOptions &>(options)) {
              ParseOptions &parse_options = nb::cast<ParseOptions &>(options);
              if (parse_options.is_parallel()) {
                stream.StartThreadPool(parse_options.num_threads);
              }
              self.ParseFromStream(stream, parse_options);
              if (parse_options.is_parallel()) {
                stream.WaitForDelayedBlock();
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
            utils::BinaryStream *stream;
            const bool has_opts = nb::isinstance<ParseOptions &>(options);
            const bool wants_no_copy = has_opts && nb::cast<ParseOptions &>(options).no_copy;
            const FileLoadMode mode =
                has_opts ? nb::cast<ParseOptions &>(options).file_load_mode : FileLoadMode::kAuto;
            if (!external_data_file.empty()) {
              EXT_ENFORCE(mode == FileLoadMode::kAuto,
                          "ParseFromFile: file_load_mode is not supported when an "
                          "external_data_file is provided (TwoFilesStream is always used).");
              stream = new utils::TwoFilesStream(file_path, external_data_file);
            } else if (mode == FileLoadMode::kMmap) {
              EXT_ENFORCE(!wants_no_copy,
                          "ParseFromFile: file_load_mode=MMAP with no_copy=True on a "
                          "single-file model is not supported because the mmap mapping is "
                          "released when ParseFromFile returns. Either set no_copy=False or "
                          "use file_load_mode=AUTO (which falls back to FileStream when "
                          "no_copy=True so inline raw_data is copied into owned buffers).");
              stream = new utils::MmapFileStream(file_path);
            } else if (mode == FileLoadMode::kFileStream ||
                       (mode == FileLoadMode::kAuto && wants_no_copy)) {
              // FileStream::CanNoCopy() is false, so no_copy=True silently falls back to
              // copying inline raw_data. Keep that behavior here so the borrowed pointers
              // exposed by an mmap-backed stream do not outlive the stream object below.
              stream = new utils::FileStream(file_path);
            } else {
              // Default path: the file is mmap'd and parsed via StringStream-derived
              // MmapFileStream. This avoids the FileStream double-buffer (4 KB read_buf_
              // on top of std::ifstream's streambuf) and the seek-to-invalidate path
              // taken on large tensor payloads, closing most of the gap with protobuf's
              // hand-tuned ParseFromIstream.
              stream = new utils::MmapFileStream(file_path);
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
            delete stream;
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
            if (nb::isinstance<SerializeOptions &>(options)) {
              self.SerializeToString(out, nb::cast<SerializeOptions &>(options));
            } else {
              SerializeOptions opts;
              self.SerializeToString(out, opts);
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
            utils::BinaryWriteStream *stream =
                external_data_file.empty()
                    ? new utils::FileWriteStream(file_path)
                    : new utils::TwoFilesWriteStream(file_path, external_data_file);
            if (nb::isinstance<SerializeOptions &>(options)) {
              SerializeProtoToStream(*to_write, *stream, nb::cast<SerializeOptions &>(options),
                                     !external_data_file.empty());
            } else {
              SerializeOptions opts;
              SerializeProtoToStream(*to_write, *stream, opts, !external_data_file.empty());
            }
            delete stream;
          },
          nb::arg("name"), nb::arg("options") = nb::none(), nb::arg("external_data_file") = "",
          "Serializes this instance into a file. If ``external_data_size`` is not empty, big "
          "weights are stored in this (depending on ``options.raw_data_threshold``). "
          "When writing to two files, temporary external-data metadata is cleared so the "
          "in-memory model stays unchanged.")
      .def(
          "__str__",
          [](cls &self) -> std::string {
            utils::PrintOptions opts;
            std::vector<std::string> rows = self.PrintToVectorString(opts);
            return utils::join_string(rows);
          },
          "Creates a printable string for this class.")
      .def(
          "CopyFrom", [](cls &self, const cls &src) { self.CopyFrom(src); },
          "Copies one instance into this one.")
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
  utils::PrintOptions opts;
  std::vector<std::string> rows = self.PrintToVectorString(opts);
  size_t compact_length = 0;
  bool has_compact_content = false;
  for (const auto &row : rows) {
    size_t first = 0;
    size_t last = row.size();
    while (first < row.size() && is_space_char(row[first])) {
      ++first;
    }
    while (last > first && is_space_char(row[last - 1])) {
      --last;
    }
    if (first == last) {
      continue;
    }
    if (has_compact_content) {
      ++compact_length;
    }
    compact_length += last - first;
    has_compact_content = true;
    if (compact_length >= max_short_repr_length) {
      return utils::join_string(rows);
    }
  }
  if (!has_compact_content) {
    return utils::join_string(rows);
  }
  std::string one_line;
  one_line.reserve(compact_length);
  for (const auto &row : rows) {
    size_t first = 0;
    size_t last = row.size();
    while (first < row.size() && is_space_char(row[first])) {
      ++first;
    }
    while (last > first && is_space_char(row[last - 1])) {
      --last;
    }
    if (first == last) {
      continue;
    }
    if (!one_line.empty()) {
      one_line += " ";
    }
    one_line.append(row, first, last - first);
  }
  return one_line;
}

template <typename T> void define_repeated_field_type(nb::class_<utils::RepeatedField<T>> &nbcls) {
  nbcls.def(nb::init<>())
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
          [](utils::RepeatedField<T> &self) {
            return nb::make_iterator(nb::type<utils::RepeatedField<T>>(), "iterator", self.begin(),
                                     self.end());
          },
          nb::keep_alive<0, 1>(), "Iterates over the elements.");
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
              self.extend(nb::cast<std::vector<T>>(iterable));
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

template <>
void define_repeated_field_type_extend(nb::class_<utils::RepeatedField<utils::String>> &nbcls) {
  nbcls
      .def(
          "append",
          [](utils::RepeatedField<utils::String> &self, const utils::String &v) {
            self.push_back(v);
          },
          nb::arg("item"), "Append one element to the list of values.")
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
          nb::arg("sequence"), "Extends the list of values.");
  nbcls_proto.def(nb::init<>())
      .def(
          "add",
          [](utils::RepeatedProtoField<T> &self) -> std::shared_ptr<T> {
            self.add();
            return self.shared_at(self.size() - 1);
          },
          "Adds an empty element.")
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

  nb::enum_<FileLoadMode>(m, "FileLoadMode",
                          "Selects the file-backed stream implementation used when parsing "
                          "a model from a file path.")
      .value("AUTO", FileLoadMode::kAuto,
             "Pick the fastest stream compatible with the other options "
             "(currently mmap, except when no_copy=True on a single-file model).")
      .value("MMAP", FileLoadMode::kMmap, "Force MmapFileStream (memory-mapped file).")
      .value("IFSTREAM", FileLoadMode::kFileStream, "Force FileStream (buffered std::ifstream).");

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
      .def_rw("file_load_mode", &ParseOptions::file_load_mode,
              "Selects the file-backed stream used when parsing a model from a path: "
              "FileLoadMode.AUTO (default) picks mmap unless no_copy=True is set on a "
              "single-file model, FileLoadMode.MMAP forces MmapFileStream, and "
              "FileLoadMode.IFSTREAM forces the buffered std::ifstream-based FileStream. "
              "Ignored when parsing from bytes or when an external_data_file is provided.");

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
      .def_rw("max_external_file_size", &SerializeOptions::max_external_file_size,
              "maximum size in bytes for one external weights file when writing external data; "
              "0 means no limit");

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
              "threshold");

  nb::class_<utils::String>(m, "String", "Simplified string with no final null character.")
      .def(nb::init<std::string>())
      .def(
          "__str__", [](const utils::String &self) -> std::string { return self.as_string(); },
          "Converts this instance into a python string.")
      .def(
          "__add__",
          [](const utils::String &self, const nb::str &s) -> std::string {
            return self.as_string() + nb::cast<std::string>(s);
          },
          "Concatenates this string and a python string.", nb::is_operator())
      .def(
          "__radd__",
          [](const utils::String &self, const nb::str &s) -> std::string {
            return nb::cast<std::string>(s) + self.as_string();
          },
          "Concatenates a python string and this string.", nb::is_operator())
      .def(
          "__repr__",
          [](const utils::String &self) -> std::string {
            return std::string("'") + self.as_string() + std::string("'");
          },
          "Representation with surrounding quotes.")
      .def(
          "__len__", [](const utils::String &self) -> int { return self.size(); },
          "Returns the length of the string.")
      .def(
          "__eq__",
          [](const utils::String &self, const std::string &s) -> int { return self == s; },
          "Compares two strings.")
      .def(
          "__eq__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> int {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self == st;
          },
          "Compares to a byte string.")
      .def(
          "__eq__",
          [](const utils::String &self, const utils::String &s) -> bool { return self == s; },
          "Compares two String instances.", nb::is_operator())
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
          "Returns the same hash as the equivalent Python str, enabling use as dict keys.");

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
  define_repeated_field_type_extend(rep_string);

  nb::enum_<OperatorStatus>(m, "OperatorStatus", nb::is_arithmetic())
      .value("EXPERIMENTAL", OperatorStatus::EXPERIMENTAL)
      .value("STABLE", OperatorStatus::STABLE)
      .export_values();

  nb::class_<Message>(m, "Message", "Message, base class for all onnx classes").def(nb::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value);
  PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(StringStringEntryProto, rep_ssentry);
  define_repeated_field_type_proto(rep_ssentry, rep_ssentry_proto);

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version);
  PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  nb_OperatorSetIdProto.def(
      "__repr__", [](OperatorSetIdProto &self) { return proto_repr_with_short_line(self); });
  DECLARE_REPEATED_FIELD_PROTO(OperatorSetIdProto, rep_osp);
  define_repeated_field_type_proto(rep_osp, rep_osp_proto);

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names);
  PYADD_PROTO_SERIALIZATION(TensorAnnotation);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value);
  PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(IntIntListEntryProto, rep_iil);
  define_repeated_field_type_proto(rep_iil, rep_iil_proto);

  PYDEFINE_PROTO(m, DeviceConfigurationProto)
      .PYFIELD_STR(DeviceConfigurationProto, name)
      .PYFIELD(DeviceConfigurationProto, num_devices)
      .PYFIELD(DeviceConfigurationProto, device);
  PYADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  PYDEFINE_PROTO(m, SimpleShardedDimProto)
      .PYFIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .PYFIELD_STR(SimpleShardedDimProto, dim_param)
      .PYFIELD(SimpleShardedDimProto, num_shards);
  PYADD_PROTO_SERIALIZATION(SimpleShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(SimpleShardedDimProto, rep_ssdp);
  define_repeated_field_type_proto(rep_ssdp, rep_ssdp_proto);

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding);
  PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardedDimProto, rep_sdp);
  define_repeated_field_type_proto(rep_sdp, rep_sdp_proto);

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim);
  PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardingSpecProto, rep_ssp);
  define_repeated_field_type_proto(rep_ssp, rep_ssp_proto);

  PYDEFINE_PROTO(m, NodeDeviceConfigurationProto)
      .PYFIELD_STR(NodeDeviceConfigurationProto, configuration_id)
      .PYFIELD(NodeDeviceConfigurationProto, sharding_spec)
      .PYFIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage);
  PYADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorShapeProto);
  PYDEFINE_SUBPROTO(nb_TensorShapeProto, TensorShapeProto, Dimension)
      .PYFIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .PYFIELD_STR(TensorShapeProto::Dimension, dim_param)
      .PYFIELD_STR(TensorShapeProto::Dimension, denotation)
      .def("__repr__",
           [](TensorShapeProto::Dimension &self) { return proto_repr_with_short_line(self); });
  PYADD_SUBPROTO_SERIALIZATION(TensorShapeProto, Dimension);
  DECLARE_REPEATED_FIELD_SUBPROTO(TensorShapeProto, Dimension, rep_tspd);
  define_repeated_field_type_proto(rep_tspd, rep_tspd_proto);
  nb_TensorShapeProto.PYFIELD(TensorShapeProto, dim);
  PYADD_PROTO_SERIALIZATION(TensorShapeProto);
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
      .def(
          "load_external_data",
          [](TensorProto &self, const std::string &base_dir) { self.LoadExternalData(base_dir); },
          nb::arg("base_dir") = std::string(),
          "Loads the raw bytes of this tensor from the external file described by its "
          "``external_data`` field into ``raw_data``. The ``external_data`` and ``data_location`` "
          "fields are preserved.");
  PYADD_PROTO_SERIALIZATION(TensorProto);
  DECLARE_REPEATED_FIELD_PROTO(TensorProto, rep_tp);
  define_repeated_field_type_proto(rep_tp, rep_tp_proto);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims);
  PYADD_PROTO_SERIALIZATION(SparseTensorProto);
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
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Tensor);

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
      .PYFIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);

  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Sequence)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Sequence);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Optional)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Optional);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Map)
      .PYFIELD(TypeProto::Map, key_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Map, value_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Map);
  nb_TypeProto.PYFIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .PYFIELD_STR(TypeProto, denotation)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, optional_type);
  PYADD_PROTO_SERIALIZATION(TypeProto);
  nb_TypeProto.def("__repr__", [](TypeProto &self) { return proto_repr_with_short_line(self); });

  PYDEFINE_PROTO(m, ValueInfoProto)
      .PYFIELD_STR(ValueInfoProto, name)
      .PYFIELD_OPTIONAL_PROTO(ValueInfoProto, type)
      .PYFIELD_STR(ValueInfoProto, doc_string)
      .PYFIELD(ValueInfoProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(ValueInfoProto);
  nb_ValueInfoProto.def("__repr__",
                        [](ValueInfoProto &self) { return proto_repr_with_short_line(self); });
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
      .PYFIELD(AttributeProto, graphs);
  PYADD_PROTO_SERIALIZATION(AttributeProto);
  nb_AttributeProto.def("__repr__",
                        [](AttributeProto &self) { return proto_repr_with_short_line(self); });
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
  nb_NodeProto.def("__repr__", [](NodeProto &self) { return proto_repr_with_short_line(self); });
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
  nb_ModelProto.def("__repr__", [](ModelProto &self) { return proto_repr_with_short_line(self); });
#ifdef ONNX_LIGHT_HAS_OPENSSL
  nb_ModelProto
      .def(
          "SerializeToEncryptedFile",
          [](ModelProto &self, const std::string &file_path, const std::string &key,
             nb::object options) {
            SerializeOptions opts;
            if (nb::isinstance<SerializeOptions>(options)) {
              opts = nb::cast<SerializeOptions>(options);
            }
            SaveEncryptedModel(self, file_path, key, opts);
          },
          nb::arg("name"), nb::arg("key"), nb::arg("options") = nb::none(),
          "Encrypts the model with AES-256-CBC (PBKDF2 key derivation) and writes it to a "
          "single binary file.  The *key* argument is a passphrase or raw bytes used to derive "
          "the AES-256 key via PBKDF2-HMAC-SHA256.")
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
          "Decrypts an ONNXCRY1 encrypted file (written by SerializeToEncryptedFile) and "
          "parses the payload into this model instance.")
      .def(
          "SerializeToEncryptedString",
          [](ModelProto &self, const std::string &key, nb::object options) {
            SerializeOptions opts;
            if (nb::isinstance<SerializeOptions>(options)) {
              opts = nb::cast<SerializeOptions>(options);
            }
            const std::string blob = SaveEncryptedModelToString(self, key, opts);
            return nb::bytes(blob.data(), blob.size());
          },
          nb::arg("key"), nb::arg("options") = nb::none(),
          "Encrypts the model with AES-256-CBC (PBKDF2 key derivation) and returns the "
          "ciphertext as a bytes object in ONNXCRY1 format.")
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
          "Decrypts an ONNXCRY1 encrypted bytes object (produced by SerializeToEncryptedString) "
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
      .def("HasField", [](const OptionalProto &self, const std::string &field_name) {
        if (self.has_tensor_value() && field_name == "tensor_value")
          return true;
        if (self.has_sparse_tensor_value() && field_name == "sparse_tensor_value")
          return true;
        if (self.has_sequence_value() && field_name == "sequence_value")
          return true;
        if (self.has_map_value() && field_name == "map_value")
          return true;
        if (self.has_optional_value() && field_name == "optional_value")
          return true;
        return false;
      });
  PYADD_PROTO_SERIALIZATION(OptionalProto);
}
