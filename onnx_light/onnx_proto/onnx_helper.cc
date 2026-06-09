// _GNU_SOURCE must be defined before any system header to expose splice(2) on Linux.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "onnx_helper.h"
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE {
bool IteratorTensorProto::next() {
  while (!positions_.empty()) {
    Position &pos = positions_.back();
    // loops over nodes
    bool break_loop = false;
    if (pos.graph->ref_node().size() > 0) {
      while (pos.node_index < static_cast<int>(pos.graph->ref_node().size())) {
        NodeProto *node = &(pos.graph->ref_node()[pos.node_index]);
        while (pos.attr_index < static_cast<int>(node->ref_attribute().size())) {
          AttributeProto &att = node->ref_attribute()[pos.attr_index];
          if (att.has_t()) {
            tp_ = &(att.ref_t());
            ++pos.attr_index;
            return true;
          } else if (att.has_g()) {
            GraphProto *subgraph = &(att.ref_g());
            // Do not switch this line with the next one, if vector needs to be resized,
            // the address of (Position&) pos changes as well.
            ++pos.attr_index;
            positions_.emplace_back(Position{subgraph});
            break_loop = true;
            break;
          }
          EXT_ENFORCE(!att.has_tensors(), "not implemented yet for attribute with tensors");
          EXT_ENFORCE(!att.has_graphs(), "not implemented yet for attribute with graphs");
          ++pos.attr_index;
        }
        if (break_loop)
          break;
        ++pos.node_index;
        pos.attr_index = 0;
      }
    }
    if (break_loop)
      continue;
    // loop over initializers
    if (pos.graph->ref_initializer().size() > 0) {
      if (pos.node_initializer_index < static_cast<int64_t>(pos.graph->ref_initializer().size())) {
        tp_ = &(pos.graph->ref_initializer()[pos.node_initializer_index]);
        ++pos.node_initializer_index;
        return true;
      }
    }
    positions_.pop_back();
  }
  return false;
}

// Rounds offset up to the nearest multiple of alignment (no-op when alignment <= 1 or offset == 0).
static inline offset_t align_up(offset_t offset, int64_t alignment) {
  if (alignment <= 1 || offset <= 0)
    return offset;
  return ((offset + alignment - 1) / alignment) * alignment;
}

static uint8_t TouchesRawDataPages(const utils::ByteSpan &raw_data) {
  static constexpr size_t kPageSize = 4096;
  const size_t n_bytes = raw_data.size();
  if (n_bytes == 0) {
    return 0;
  }
  const volatile uint8_t *data = raw_data.data();
  uint8_t checksum = 0;
  for (size_t i = 0; i < n_bytes; i += kPageSize) {
    checksum = static_cast<uint8_t>(checksum + data[i]);
  }
  checksum = static_cast<uint8_t>(checksum + data[n_bytes - 1]);
  return checksum;
}

static uint64_t TouchesAllModelRawDataPages(ModelProto &model) {
  uint64_t checksum = 0;
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (it->has_raw_data()) {
      checksum += TouchesRawDataPages(it->ref_raw_data());
    }
  }
  return checksum;
}

offset_t PopulateExternalData(ModelProto &model, size_t threshold,
                              const std::string &external_data_location,
                              bool use_external_data_location, int64_t max_external_file_size,
                              int64_t alignment) {
  onnx_light_helpers::ValidateAlignmentOption(alignment, "SerializeOptions.alignment");
  offset_t offset = 0;
  int64_t file_index = 0;
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (it->has_raw_data() && it->raw_data_.size() >= threshold) {
      if (use_external_data_location && it->has_data_location() &&
          it->ref_data_location() == TensorProto::DataLocation::EXTERNAL) {
        continue;
      }
      // Align the current offset before placing this tensor.
      offset = align_up(offset, alignment);
      if (max_external_file_size > 0 && offset > 0 &&
          offset + static_cast<offset_t>(it->raw_data_.size()) > max_external_file_size) {
        ++file_index;
        offset = 0;
      }
      std::string location = external_data_location;
      if (file_index > 0) {
        location.append(".").append(std::to_string(file_index));
      }
      EXT_ENFORCE(!it->has_external_data(), "External data should not be set already.");
      EXT_ENFORCE(!it->has_data_location() ||
                      it->ref_data_location() == TensorProto::DataLocation::DEFAULT,
                  "External data should not be set already.");
      it->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
      StringStringEntryProto *loc = it->add_external_data();
      loc->set_key("location");
      loc->set_value(location);
      StringStringEntryProto *off = it->add_external_data();
      off->set_key("offset");
      off->set_value(onnx_light_helpers::MakeString(offset));
      StringStringEntryProto *size = it->add_external_data();
      size->set_key("length");
      size->set_value(std::to_string(it->raw_data_.size()));
      offset += it->raw_data_.size();
    }
  }
  return offset;
}

void ClearExternalData(ModelProto &model) {
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (it->has_external_data()) {
      EXT_ENFORCE(it->has_raw_data(), "raw_data is empty, external data should not be removed.");
      it->clr_external_data();
      it->reset_data_location();
    }
  }
}

namespace {

// Generates a UUIDv1-like 8-4-4-4-12 hex string without external dependencies.
// Used as the fallback external_data filename when the caller does not pass one.
std::string MakeUuidFilename() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  uint64_t a = dist(gen);
  uint64_t b = dist(gen);
  std::array<char, 64> buf{};
  std::snprintf(buf.data(), buf.size(), "%08x-%04x-%04x-%04x-%012llx.data",
                static_cast<uint32_t>(a >> 32), static_cast<uint16_t>(a >> 16),
                static_cast<uint16_t>(a), static_cast<uint16_t>(b >> 48),
                static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFULL));
  return std::string(buf.data());
}

// Returns true when the filename contains no path-separator or otherwise illegal
// characters; mirrors the upstream onnx.external_data_helper._sanitize_path check.
bool IsValidExternalDataFilename(const std::string &name) {
  if (name.empty()) {
    return false;
  }
  for (char c : name) {
    switch (c) {
    case '<':
    case '>':
    case ':':
    case ';':
    case ',':
    case '?':
    case '"':
    case '*':
    case '|':
    case '/':
    case '\\':
      return false;
    default:
      break;
    }
  }
  return true;
}

// Sets *tensor* as EXTERNAL with a single ``location`` entry, mirroring
// onnx.external_data_helper.set_external_data.  The tensor must currently
// carry inline raw_data; the bytes are left untouched so they can be written
// out by a subsequent serialization call.
void SetExternalDataLocation(TensorProto &tensor, const std::string &location) {
  EXT_ENFORCE(tensor.has_raw_data(), "Tensor '", tensor.ref_name().as_string(),
              "' does not have raw_data. Cannot set external data for this tensor.");
  tensor.clr_external_data();
  tensor.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  StringStringEntryProto *loc = tensor.add_external_data();
  loc->set_key("location");
  loc->set_value(location);
}

// Walks every initializer tensor in *graph* and recursively in any subgraph
// nested in node attributes (AttributeProto.g / AttributeProto.graphs),
// invoking ``fn`` on each.
template <typename F> void ForEachInitializerTensor(GraphProto &graph, F &&fn) {
  for (int i = 0; i < graph.ref_initializer().size(); ++i) {
    fn(graph.ref_initializer()[i]);
  }
  for (int i = 0; i < graph.ref_node().size(); ++i) {
    NodeProto &node = graph.ref_node()[i];
    for (int j = 0; j < node.ref_attribute().size(); ++j) {
      AttributeProto &attr = node.ref_attribute()[j];
      if (attr.has_g()) {
        ForEachInitializerTensor(attr.ref_g(), fn);
      }
      if (attr.has_graphs()) {
        for (int k = 0; k < attr.ref_graphs().size(); ++k) {
          ForEachInitializerTensor(attr.ref_graphs()[k], fn);
        }
      }
    }
  }
}

// Walks every TensorProto stored in node attributes (AttributeProto.t and
// AttributeProto.tensors) of every node in *nodes*, recursing into any
// subgraph nested in those attributes.
template <typename Nodes, typename F> void ForEachAttributeTensorInNodes(Nodes &nodes, F &&fn);

template <typename F> void ForEachAttributeTensorInGraph(GraphProto &graph, F &&fn) {
  ForEachAttributeTensorInNodes(graph.ref_node(), fn);
}

template <typename Nodes, typename F> void ForEachAttributeTensorInNodes(Nodes &nodes, F &&fn) {
  for (int i = 0; i < nodes.size(); ++i) {
    NodeProto &node = nodes[i];
    for (int j = 0; j < node.ref_attribute().size(); ++j) {
      AttributeProto &attr = node.ref_attribute()[j];
      if (attr.has_t()) {
        fn(attr.ref_t());
      }
      if (attr.has_tensors()) {
        for (int k = 0; k < attr.ref_tensors().size(); ++k) {
          fn(attr.ref_tensors()[k]);
        }
      }
      if (attr.has_g()) {
        ForEachAttributeTensorInGraph(attr.ref_g(), fn);
      }
      if (attr.has_graphs()) {
        for (int k = 0; k < attr.ref_graphs().size(); ++k) {
          ForEachAttributeTensorInGraph(attr.ref_graphs()[k], fn);
        }
      }
    }
  }
}

} // namespace

void ConvertModelToExternalData(ModelProto &model, bool all_tensors_to_one_file,
                                const std::string &location, size_t size_threshold,
                                bool convert_attribute) {
  // Decide the target filename when every tensor goes to the same file.
  std::string single_file_name;
  if (all_tensors_to_one_file) {
    if (location.empty()) {
      single_file_name = MakeUuidFilename();
    } else {
      if (std::filesystem::exists(location)) {
        throw ExternalDataLocationExistsError(
            onnx_light_helpers::MakeString("External data file exists in ", location, "."));
      }
      if (std::filesystem::path(location).is_absolute()) {
        EXT_THROW_INVALID("location must be a relative path that is relative to the model path.");
      }
      single_file_name = location;
    }
  }

  auto handle_tensor = [&](TensorProto &tensor) {
    if (!tensor.has_raw_data()) {
      return;
    }
    if (tensor.ref_raw_data().size() < size_threshold) {
      return;
    }
    if (all_tensors_to_one_file) {
      SetExternalDataLocation(tensor, single_file_name);
    } else {
      std::string tensor_location = tensor.ref_name().as_string();
      if (!IsValidExternalDataFilename(tensor_location)) {
        tensor_location = MakeUuidFilename();
      }
      SetExternalDataLocation(tensor, tensor_location);
    }
  };

  if (model.has_graph()) {
    ForEachInitializerTensor(model.ref_graph(), handle_tensor);
  }
  if (convert_attribute) {
    if (model.has_graph()) {
      ForEachAttributeTensorInGraph(model.ref_graph(), handle_tensor);
    }
    if (model.has_functions()) {
      for (int i = 0; i < model.ref_functions().size(); ++i) {
        FunctionProto &fn = model.ref_functions()[i];
        ForEachAttributeTensorInNodes(fn.ref_node(), handle_tensor);
      }
    }
  }
}

void LoadExternalDataForModel(ModelProto &model, const std::string &base_dir) {
  auto handle_tensor = [&](TensorProto &tensor) {
    if (!tensor.has_data_location() ||
        tensor.ref_data_location() != TensorProto::DataLocation::EXTERNAL) {
      return;
    }
    tensor.LoadExternalData(base_dir);
    tensor.ref_data_location() = TensorProto::DataLocation::DEFAULT;
    tensor.clr_external_data();
  };

  if (model.has_graph()) {
    ForEachInitializerTensor(model.ref_graph(), handle_tensor);
    ForEachAttributeTensorInGraph(model.ref_graph(), handle_tensor);
  }
  if (model.has_functions()) {
    for (int i = 0; i < model.ref_functions().size(); ++i) {
      FunctionProto &fn = model.ref_functions()[i];
      ForEachAttributeTensorInNodes(fn.ref_node(), handle_tensor);
    }
  }
}

namespace {

// Parses an int64_t encoded as decimal text inside external_data entries (offset/length/size).
int64_t ParseExternalDataInt64(const utils::String &value, const char *key) {
  int64_t out = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto parsed = std::from_chars(begin, end, out);
  EXT_ENFORCE(parsed.ec == std::errc() && parsed.ptr == end,
              "AlignExternalDataStreaming: unable to parse external_data '", key,
              "' as int64, value='", value.as_string(true), "'.");
  return out;
}

// Reads location/offset/length out of a tensor's external_data; length is optional and falls
// back to the alias 'size' before defaulting to -1 (caller must enforce).
void ReadExternalDataEntries(const TensorProto &tensor, std::string &location, int64_t &offset,
                             int64_t &length) {
  location.clear();
  offset = 0;
  length = -1;
  bool has_offset = false;
  for (int i = 0; i < tensor.ref_external_data().size(); ++i) {
    const StringStringEntryProto &entry = tensor.ref_external_data()[i];
    const utils::String &key = entry.ref_key();
    if (key == "location") {
      location = entry.ref_value().as_string();
    } else if (key == "offset") {
      offset = ParseExternalDataInt64(entry.ref_value(), "offset");
      has_offset = true;
    } else if (key == "length" || key == "size") {
      length = ParseExternalDataInt64(entry.ref_value(), key.data());
    }
  }
  EXT_ENFORCE(!location.empty(), "AlignExternalDataStreaming: tensor '",
              tensor.ref_name().as_string(), "' has no external_data.location.");
  EXT_ENFORCE(has_offset || offset == 0, "AlignExternalDataStreaming: tensor '",
              tensor.ref_name().as_string(), "' has invalid external_data.offset.");
  EXT_ENFORCE(length >= 0, "AlignExternalDataStreaming: tensor '", tensor.ref_name().as_string(),
              "' has no external_data.length.");
}

// Rewrites the existing external_data entries to point at (new_location, new_offset, length).
// Preserves the 'checksum' entry (if any) since the bytes content is unchanged.
void RewriteExternalDataEntries(TensorProto &tensor, const std::string &new_location,
                                int64_t new_offset, int64_t length) {
  std::string checksum;
  bool has_checksum = false;
  for (int i = 0; i < tensor.ref_external_data().size(); ++i) {
    const StringStringEntryProto &entry = tensor.ref_external_data()[i];
    if (entry.ref_key() == "checksum") {
      checksum = entry.ref_value().as_string();
      has_checksum = true;
      break;
    }
  }
  tensor.clr_external_data();
  StringStringEntryProto *loc = tensor.add_external_data();
  loc->set_key("location");
  loc->set_value(new_location);
  StringStringEntryProto *off = tensor.add_external_data();
  off->set_key("offset");
  off->set_value(onnx_light_helpers::MakeString(new_offset));
  StringStringEntryProto *len = tensor.add_external_data();
  len->set_key("length");
  len->set_value(onnx_light_helpers::MakeString(length));
  if (has_checksum) {
    StringStringEntryProto *ck = tensor.add_external_data();
    ck->set_key("checksum");
    ck->set_value(checksum);
  }
}

// Streams n_bytes from src starting at src_offset into dst at its current position.
// On Linux uses splice(2) via a kernel pipe (zero-copy in kernel space); on other platforms
// falls back to a buffered read/write loop. `buffer.size()` bounds the per-iteration transfer.
void StreamCopyFileRange(int src_fd, int64_t src_offset, int64_t n_bytes, int dst_fd,
                         std::vector<char> &buffer) {
  EXT_ENFORCE(n_bytes >= 0, "StreamCopyFileRange: n_bytes must be >= 0, got ", n_bytes, ".");
  if (n_bytes == 0)
    return;
  EXT_ENFORCE(!buffer.empty(), "StreamCopyFileRange: chunk buffer must be non-empty.");
  const size_t chunk = buffer.size();
#if defined(__linux__)
  // Zero-copy path: src_fd -> pipe -> dst_fd, entirely in kernel space.
  int pipefd[2];
  EXT_ENFORCE(::pipe(pipefd) == 0, "StreamCopyFileRange: pipe() failed (errno=", errno, ": ",
              std::strerror(errno), ").");
  // RAII pipe closer so we never leak fds on exception.
  struct PipeGuard {
    int *p;
    ~PipeGuard() {
      if (p[0] >= 0)
        ::close(p[0]);
      if (p[1] >= 0)
        ::close(p[1]);
    }
  } pg{pipefd};
  loff_t in_off = static_cast<loff_t>(src_offset);
  int64_t remaining = n_bytes;
  while (remaining > 0) {
    const size_t to_xfer =
        static_cast<size_t>(std::min<int64_t>(remaining, static_cast<int64_t>(chunk)));
    ssize_t r =
        ::splice(src_fd, &in_off, pipefd[1], nullptr, to_xfer, SPLICE_F_MOVE | SPLICE_F_MORE);
    if (r < 0 && errno == EINTR)
      continue;
    EXT_ENFORCE(r > 0, "StreamCopyFileRange: splice(src->pipe) failed at offset ",
                src_offset + (n_bytes - remaining), " (errno=", errno, ": ", std::strerror(errno),
                ", returned ", r, ").");
    ssize_t to_drain = r;
    while (to_drain > 0) {
      ssize_t w = ::splice(pipefd[0], nullptr, dst_fd, nullptr, static_cast<size_t>(to_drain),
                           SPLICE_F_MOVE | SPLICE_F_MORE);
      if (w < 0 && errno == EINTR)
        continue;
      EXT_ENFORCE(w > 0, "StreamCopyFileRange: splice(pipe->dst) failed (errno=", errno, ": ",
                  std::strerror(errno), ", returned ", w, ").");
      to_drain -= w;
    }
    remaining -= r;
  }
#else
  // Portable fallback: pread/read + write loop through the heap buffer.
#if defined(_WIN32)
  const __int64 seeked = ::_lseeki64(src_fd, static_cast<__int64>(src_offset), SEEK_SET);
  EXT_ENFORCE(seeked == static_cast<__int64>(src_offset),
              "StreamCopyFileRange: _lseeki64 failed (errno=", errno, ": ", std::strerror(errno),
              ").");
#else
  const off_t seeked = ::lseek(src_fd, static_cast<off_t>(src_offset), SEEK_SET);
  EXT_ENFORCE(seeked == static_cast<off_t>(src_offset),
              "StreamCopyFileRange: lseek failed (errno=", errno, ": ", std::strerror(errno), ").");
#endif
  int64_t remaining = n_bytes;
  while (remaining > 0) {
    const size_t to_read =
        static_cast<size_t>(std::min<int64_t>(remaining, static_cast<int64_t>(chunk)));
#if defined(_WIN32)
    int r = ::_read(src_fd, buffer.data(), static_cast<unsigned>(to_read));
#else
    ssize_t r = ::read(src_fd, buffer.data(), to_read);
    if (r < 0 && errno == EINTR)
      continue;
#endif
    EXT_ENFORCE(r > 0 && static_cast<size_t>(r) == to_read,
                "StreamCopyFileRange: short read from source weights file (asked for ", to_read,
                ", got ", r, ", at offset ", src_offset + (n_bytes - remaining), ", errno=", errno,
                ": ", std::strerror(errno), ").");
    const char *p = buffer.data();
    size_t left = static_cast<size_t>(r);
    while (left > 0) {
#if defined(_WIN32)
      int w = ::_write(dst_fd, p, static_cast<unsigned>(left));
#else
      ssize_t w = ::write(dst_fd, p, left);
      if (w < 0 && errno == EINTR)
        continue;
#endif
      EXT_ENFORCE(w > 0, "StreamCopyFileRange: failed to write to destination (errno=", errno, ": ",
                  std::strerror(errno), ").");
      p += w;
      left -= static_cast<size_t>(w);
    }
    remaining -= r;
  }
#endif
}

// Writes `n` zero bytes to dst_fd; used to pad up to alignment between tensors.
void WriteZeros(int dst_fd, int64_t n) {
  static constexpr size_t kZeroBufSize = 4096;
  static const std::array<char, kZeroBufSize> kZeros{}; // value-initialized → zero-filled
  while (n > 0) {
    const size_t to_write = static_cast<size_t>(std::min<int64_t>(n, kZeroBufSize));
    const char *p = kZeros.data();
    size_t left = to_write;
    while (left > 0) {
#if defined(_WIN32)
      int w = ::_write(dst_fd, p, static_cast<unsigned>(left));
#else
      ssize_t w = ::write(dst_fd, p, left);
      if (w < 0 && errno == EINTR)
        continue;
#endif
      EXT_ENFORCE(w > 0,
                  "AlignExternalDataStreaming: failed to write padding to destination (errno=",
                  errno, ": ", std::strerror(errno), ").");
      p += w;
      left -= static_cast<size_t>(w);
    }
    n -= to_write;
  }
}

// RAII wrapper around a POSIX-style file descriptor (auto-closes on destruction).
class ScopedFd {
public:
  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd &) = delete;
  ScopedFd &operator=(const ScopedFd &) = delete;
  ScopedFd(ScopedFd &&other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  ScopedFd &operator=(ScopedFd &&other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  ~ScopedFd() { reset(); }
  int get() const { return fd_; }
  void reset() {
    if (fd_ >= 0) {
#if defined(_WIN32)
      ::_close(fd_);
#else
      ::close(fd_);
#endif
      fd_ = -1;
    }
  }

private:
  int fd_ = -1;
};

int OpenForRead(const std::string &path) {
#if defined(_WIN32)
  return ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
  return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

int OpenForWriteTrunc(const std::string &path) {
#if defined(_WIN32)
  return ::_open(path.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
  return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
#endif
}

} // namespace

offset_t AlignExternalDataStreaming(const std::string &src_onnx_path,
                                    const std::string &dst_onnx_path,
                                    const std::string &dst_weights_path, int64_t alignment,
                                    int64_t chunk_size) {
  onnx_light_helpers::ValidateAlignmentOption(alignment, "AlignExternalDataStreaming.alignment");
  EXT_ENFORCE(alignment >= 1, "AlignExternalDataStreaming: alignment must be >= 1, got ", alignment,
              ".");
  EXT_ENFORCE(chunk_size > 0, "AlignExternalDataStreaming: chunk_size must be > 0, got ",
              chunk_size, ".");

  const std::filesystem::path src_path(src_onnx_path);
  const std::filesystem::path src_dir = src_path.parent_path();
  const std::filesystem::path dst_path(dst_onnx_path);
  const std::filesystem::path dst_dir = dst_path.parent_path();
  const std::filesystem::path dst_weights_full(dst_weights_path);

  // Location to record in external_data: path of dst_weights relative to dst_onnx's directory
  // (matches how SerializeModelProtoToStream computes locations for TwoFilesWriteStream).
  std::filesystem::path stored_location =
      dst_dir.empty() ? dst_weights_full : std::filesystem::relative(dst_weights_full, dst_dir);
  if (stored_location.empty()) {
    stored_location = dst_weights_full;
  }
  const std::string stored_location_str = stored_location.string();

  // 1) Parse the source .onnx with skip_raw_data=true so weights bytes are never loaded.
  ModelProto model;
  {
    utils::FileStream rstream(src_onnx_path);
    ParseOptions ropts;
    ropts.skip_raw_data = true;
    ropts.raw_data_threshold = 0; // skip all raw_data, regardless of size
    // Important: do not auto-clear external_data, we need it to drive the copy.
    ParseModelProtoFromStream(model, rstream, ropts, /*clear_external_data=*/false);
  }

  // 2) Stream-copy external tensor bytes from their source files into a single aligned dst file,
  //    updating each tensor's external_data entries in-place.
  ScopedFd out_fd(OpenForWriteTrunc(dst_weights_path));
  EXT_ENFORCE(out_fd.get() >= 0,
              "AlignExternalDataStreaming: cannot open destination weights file '",
              dst_weights_path, "' for writing (errno=", errno, ": ", std::strerror(errno), ").");

  // Cache one fd per source weights location so we don't reopen for every tensor.
  std::unordered_map<std::string, ScopedFd> src_fds;
  std::vector<char> buffer(static_cast<size_t>(chunk_size));
  offset_t current_offset = 0;

  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    TensorProto &tensor = *it;
    const bool has_inline_raw = tensor.has_raw_data() && tensor.raw_data_.size() > 0;
    const bool is_external = tensor.has_data_location() &&
                             tensor.ref_data_location() == TensorProto::DataLocation::EXTERNAL &&
                             tensor.has_external_data();
    EXT_ENFORCE(!has_inline_raw || !is_external, "AlignExternalDataStreaming: tensor '",
                tensor.ref_name().as_string(),
                "' has both inline raw_data and external_data; this is not supported.");
    if (!is_external) {
      // Inline tensors are preserved as-is in the destination .onnx. If they had large
      // inline raw_data, ParseModelProtoFromStream with skip_raw_data=true would have
      // dropped the bytes; detect and refuse such cases so we never silently corrupt data.
      EXT_ENFORCE(!tensor.has_data_location() ||
                      tensor.ref_data_location() == TensorProto::DataLocation::DEFAULT,
                  "AlignExternalDataStreaming: tensor '", tensor.ref_name().as_string(),
                  "' is marked EXTERNAL but has no external_data entries.");
      continue;
    }

    std::string src_location;
    int64_t src_offset = 0;
    int64_t length = 0;
    ReadExternalDataEntries(tensor, src_location, src_offset, length);

    // Resolve source weights file path relative to src .onnx directory.
    std::filesystem::path src_weights_path = src_location;
    if (!src_weights_path.is_absolute() && !src_dir.empty()) {
      src_weights_path = src_dir / src_location;
    }
    const std::string src_weights_key = src_weights_path.string();

    auto stream_it = src_fds.find(src_weights_key);
    if (stream_it == src_fds.end()) {
      int fd = OpenForRead(src_weights_key);
      EXT_ENFORCE(fd >= 0, "AlignExternalDataStreaming: cannot open source weights file '",
                  src_weights_key, "' for tensor '", tensor.ref_name().as_string(),
                  "' (errno=", errno, ": ", std::strerror(errno), ").");
      stream_it = src_fds.emplace(src_weights_key, ScopedFd(fd)).first;
    }

    // Pad the destination file up to the next aligned offset (zero bytes).
    const offset_t aligned_offset =
        (current_offset == 0 || alignment <= 1)
            ? current_offset
            : ((current_offset + alignment - 1) / alignment) * alignment;
    if (aligned_offset > current_offset) {
      WriteZeros(out_fd.get(), aligned_offset - current_offset);
      current_offset = aligned_offset;
    }

    // Stream-copy the tensor bytes (splice(2) on Linux, buffered read/write elsewhere).
    StreamCopyFileRange(stream_it->second.get(), src_offset, length, out_fd.get(), buffer);

    // Update metadata to point at the new file/offset/length.
    RewriteExternalDataEntries(tensor, stored_location_str, current_offset, length);
    current_offset += length;
  }

#if defined(_WIN32)
  EXT_ENFORCE(::_commit(out_fd.get()) == 0,
              "AlignExternalDataStreaming: failed to flush destination weights file (errno=", errno,
              ": ", std::strerror(errno), ").");
#else
  // fsync is not strictly needed (close will flush), but matches the prior flush+close intent.
  // We intentionally do not fail on fsync errors that may occur on tmpfs in CI.
  ::fsync(out_fd.get());
#endif
  out_fd.reset();

  // 3) Persist the updated proto to dst_onnx. Use FileWriteStream (single-file) so external_data
  //    entries are written as-is and no new weights file is generated.
  {
    utils::FileWriteStream wstream(dst_onnx_path);
    SerializeOptions wopts;
    SerializeProtoToStream(model, wstream, wopts);
  }

  return current_offset;
}

offset_t SaveModelWithSharedExternalData(ModelProto &model, const std::string &dst_onnx_path,
                                         const SerializeOptions &options) {
  const int64_t alignment = options.alignment;
  onnx_light_helpers::ValidateAlignmentOption(alignment,
                                              "SaveModelWithSharedExternalData.alignment");

  namespace fs = std::filesystem;
  const fs::path dst_path(dst_onnx_path);
  const fs::path dst_dir = dst_path.parent_path();

  // Secondary weights file is always next to dst_onnx_path with a ".data" suffix.
  const std::string dst_weights_path = dst_onnx_path + ".data";
  const fs::path dst_weights_full(dst_weights_path);

  // Location to record for new initializers: path of the secondary weights file relative to
  // dst_onnx_path's parent directory (i.e. just its filename).
  fs::path stored_new_location =
      dst_dir.empty() ? dst_weights_full : fs::relative(dst_weights_full, dst_dir);
  if (stored_new_location.empty()) {
    stored_new_location = dst_weights_full;
  }
  const std::string stored_new_location_str = stored_new_location.string();

  // Lazily open the destination weights file only when there is at least one new initializer.
  ScopedFd out_fd;
  bool out_open = false;
  offset_t current_offset = 0;

  auto ensure_out_open = [&]() {
    if (!out_open) {
      out_fd = ScopedFd(OpenForWriteTrunc(dst_weights_path));
      EXT_ENFORCE(out_fd.get() >= 0,
                  "SaveModelWithSharedExternalData: cannot open destination weights file '",
                  dst_weights_path, "' for writing (errno=", errno, ": ", std::strerror(errno),
                  ").");
      out_open = true;
    }
  };

  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    TensorProto &tensor = *it;
    const bool has_inline_raw = tensor.has_raw_data() && tensor.raw_data_.size() > 0;
    const bool is_external = tensor.has_data_location() &&
                             tensor.ref_data_location() == TensorProto::DataLocation::EXTERNAL &&
                             tensor.has_external_data();
    EXT_ENFORCE(!has_inline_raw || !is_external, "SaveModelWithSharedExternalData: tensor '",
                tensor.ref_name().as_string(),
                "' has both inline raw_data and external_data; this is not supported.");

    if (is_external) {
      // Reused initializer: its external_data entries belong to a previously saved model and
      // are kept as-is. The caller is responsible for the recorded location remaining
      // resolvable relative to dst_onnx_path's parent directory.
      continue;
    }

    if (!has_inline_raw) {
      // Nothing to externalize (e.g. tensors using typed *_data fields, or empty raw_data).
      EXT_ENFORCE(!tensor.has_data_location() ||
                      tensor.ref_data_location() == TensorProto::DataLocation::DEFAULT,
                  "SaveModelWithSharedExternalData: tensor '", tensor.ref_name().as_string(),
                  "' is marked EXTERNAL but has no external_data entries.");
      continue;
    }

    // New initializer: write its raw_data to dst_weights_path at an aligned offset.
    ensure_out_open();
    const offset_t aligned_offset =
        (current_offset == 0 || alignment <= 1)
            ? current_offset
            : ((current_offset + alignment - 1) / alignment) * alignment;
    if (aligned_offset > current_offset) {
      WriteZeros(out_fd.get(), aligned_offset - current_offset);
      current_offset = aligned_offset;
    }

    const size_t n_bytes = tensor.raw_data_.size();
    const char *p = reinterpret_cast<const char *>(tensor.raw_data_.data());
    size_t left = n_bytes;
    while (left > 0) {
#if defined(_WIN32)
      int w = ::_write(out_fd.get(), p, static_cast<unsigned>(left));
#else
      ssize_t w = ::write(out_fd.get(), p, left);
      if (w < 0 && errno == EINTR)
        continue;
#endif
      EXT_ENFORCE(w > 0, "SaveModelWithSharedExternalData: failed to write tensor '",
                  tensor.ref_name().as_string(), "' to destination weights file (errno=", errno,
                  ": ", std::strerror(errno), ").");
      p += w;
      left -= static_cast<size_t>(w);
    }

    // Promote the tensor to EXTERNAL pointing at dst_weights_path; drop the inline bytes so
    // the upcoming serialization (single-file FileWriteStream) does not write them inline.
    tensor.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
    tensor.clr_external_data();
    StringStringEntryProto *loc = tensor.add_external_data();
    loc->set_key("location");
    loc->set_value(stored_new_location_str);
    StringStringEntryProto *off = tensor.add_external_data();
    off->set_key("offset");
    off->set_value(onnx_light_helpers::MakeString(current_offset));
    StringStringEntryProto *len = tensor.add_external_data();
    len->set_key("length");
    len->set_value(onnx_light_helpers::MakeString(static_cast<int64_t>(n_bytes)));
    tensor.raw_data_.clear();

    current_offset += static_cast<offset_t>(n_bytes);
  }

  if (out_open) {
#if defined(_WIN32)
    EXT_ENFORCE(::_commit(out_fd.get()) == 0,
                "SaveModelWithSharedExternalData: failed to flush destination weights file (errno=",
                errno, ": ", std::strerror(errno), ").");
#else
    ::fsync(out_fd.get());
#endif
    out_fd.reset();
  }

  // Persist the updated proto to dst_onnx_path as a single .onnx file. external_data entries
  // (both for reused and newly externalized initializers) are written as-is.
  {
    utils::FileWriteStream wstream(dst_onnx_path);
    SerializeOptions wopts;
    SerializeProtoToStream(model, wstream, wopts);
  }

  return current_offset;
}

std::shared_ptr<uint8_t[]> ConsolidateTensorsToBuffer(ModelProto &model,
                                                      const TensorBufferOptions &opts) {
  onnx_light_helpers::ValidateAlignmentOption(opts.alignment, "TensorBufferOptions.alignment");

  const size_t threshold = static_cast<size_t>(std::max<int64_t>(opts.raw_data_threshold, 0));

  // First pass: compute per-tensor offsets and the total buffer size.
  struct TensorSlice {
    TensorProto *tensor;
    offset_t offset;
    size_t size;
  };
  std::vector<TensorSlice> slices;
  offset_t total_size = 0;

  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (!it->has_raw_data())
      continue;
    const size_t sz = it->ref_raw_data().size();
    if (sz < threshold)
      continue;
    total_size = align_up(total_size, opts.alignment);
    slices.push_back({&(*it), total_size, sz});
    total_size += static_cast<offset_t>(sz);
  }

  if (slices.empty() || total_size == 0)
    return nullptr;

  // Allocate a single buffer, with alignment headroom so that offset 0 can be aligned.
  const size_t alloc_size = static_cast<size_t>(total_size) +
                            (opts.alignment > 1 ? static_cast<size_t>(opts.alignment - 1) : 0);
  std::shared_ptr<uint8_t[]> storage(new uint8_t[alloc_size]);

  // Find the aligned start within the buffer.
  uint8_t *aligned_start = storage.get();
  if (opts.alignment > 1) {
    void *vptr = storage.get();
    size_t space = alloc_size;
    void *aligned = std::align(static_cast<size_t>(opts.alignment), static_cast<size_t>(total_size),
                               vptr, space);
    EXT_ENFORCE(aligned != nullptr, "ConsolidateTensorsToBuffer: failed to align buffer to ",
                opts.alignment, " bytes.");
    aligned_start = static_cast<uint8_t *>(aligned);
  }

  // Owner token shared by all borrowing tensors; the buffer stays alive as long as
  // at least one tensor (or the returned handle) holds a reference.
  auto owner = std::static_pointer_cast<void>(storage);

  // Second pass: copy raw_data into the buffer and rebind each tensor as a borrowed slice.
  for (const TensorSlice &s : slices) {
    uint8_t *dest = aligned_start + s.offset;
    std::memcpy(dest, s.tensor->ref_raw_data().data(), s.size);
    s.tensor->ref_raw_data().assign_borrowed(dest, s.size, owner);
  }

  return storage;
}

void SerializeModelProtoToStream(ModelProto &model, utils::BinaryWriteStream &stream,
                                 SerializeOptions &options, bool clear_external_data) {
  if (options.is_parallel())
    stream.StartThreadPool(options.num_threads);
  if (stream.ExternalWeights()) {
    utils::TwoFilesWriteStream &two_stream = dynamic_cast<utils::TwoFilesWriteStream &>(stream);
    std::filesystem::path parent_path = two_stream.file_path();
    parent_path = parent_path.parent_path();
    std::filesystem::path weight_path = two_stream.weights_file_path();
    weight_path = std::filesystem::relative(weight_path, parent_path);
    if (weight_path.empty()) {
      // If the relative path is empty, it means the weight file is in the same directory as the
      // model.
      weight_path = two_stream.weights_file_path();
    }
    offset_t total_external_size = PopulateExternalData(
        model, options.raw_data_threshold, weight_path.string(), options.use_external_data_location,
        options.max_external_file_size, options.alignment);
    if (options.is_parallel() && total_external_size > 0 && options.max_external_file_size <= 0) {
      two_stream.pre_allocate_weights(total_external_size);
      two_stream.StartWriteThreadPool(options.num_threads);
    }
  }
  model.SerializeToStream(stream, options);
  if (options.is_parallel())
    stream.WaitForDelayedBlock();
  if (stream.ExternalWeights()) {
    utils::TwoFilesWriteStream &two_stream = dynamic_cast<utils::TwoFilesWriteStream &>(stream);
    two_stream.WaitForWriteCompletion();
    // Flush the buffered main .onnx structure to the primary file in one write.
    // This keeps the two output streams separate: all tensor weight bytes have
    // been written to the weights file already; the main .onnx bytes were
    // accumulated in main_buf_ and are now committed together.
    two_stream.FlushMainToFile();
  }
  if (stream.ExternalWeights() && clear_external_data)
    ClearExternalData(model);
}

void ParseModelProtoFromStream(ModelProto &model, utils::BinaryStream &stream,
                               ParseOptions &options, bool clear_external_data) {
  // Mirror SerializeModelProtoToStream: start the thread pool when requested and
  // wait for all delayed reads once parsing is complete.
  if (options.is_parallel() && !stream.HasParallelizationStarted())
    stream.StartThreadPool(options.num_threads);
  if (stream.ExternalWeights()) {
    // no_copy ownership model:
    // - External-data tensors borrow slices from TwoFilesStream shared weights buffers.
    //   TensorProto::raw_data stores a shared_ptr owner (ByteSpan::owner_) so those
    //   mmap-backed buffers stay alive as long as the parsed model keeps borrowed tensors.
    // - Inline protobuf raw_data borrowed from an input bytes buffer is different:
    //   the caller must keep the original bytes object alive for the model lifetime.
    utils::TwoFilesStream &two_stream = dynamic_cast<utils::TwoFilesStream &>(stream);
    std::filesystem::path parent_path = two_stream.file_path();
    parent_path = parent_path.parent_path();
    std::filesystem::path weight_path = two_stream.weights_file_path();
    weight_path = std::filesystem::relative(weight_path, parent_path);
    if (weight_path.empty()) {
      // If the relative path is empty, it means the weight file is in the same directory as the
      // model.
      weight_path = two_stream.weights_file_path();
    }
  }
  model.ParseFromStream(stream, options);
  if (options.is_parallel())
    stream.WaitForDelayedBlock();
  if (options._touch_raw_data_pages) {
    (void)TouchesAllModelRawDataPages(model);
  }
  if (stream.ExternalWeights() && clear_external_data)
    ClearExternalData(model);
}

// Extracts the integer values of ``tensor_proto`` into ``out``. Reads
// from the type-specific repeated field when available, otherwise
// falls back to ``raw_data`` (little-endian, as required by the ONNX
// spec). Only the integer data types accepted by
// :cpp:func:`IsIntegerTensorType` are supported; ``out`` is left
// untouched and the function returns ``false`` when the underlying
// data is not present in any recognised location.
bool ReadIntegerValues(const TensorProto &tensor_proto, std::vector<int64_t> &out) {
  const auto dtype = tensor_proto.data_type();
  out.clear();

  // Type-specific storage takes precedence over raw_data when populated.
  if (dtype == TensorProto::DataType::INT64 && tensor_proto.int64_data().size() > 0) {
    out.reserve(tensor_proto.int64_data().size());
    for (int i = 0; i < tensor_proto.int64_data().size(); ++i) {
      out.push_back(tensor_proto.int64_data()[i]);
    }
    return true;
  }
  if ((dtype == TensorProto::DataType::INT32 || dtype == TensorProto::DataType::INT16 ||
       dtype == TensorProto::DataType::INT8 || dtype == TensorProto::DataType::UINT16 ||
       dtype == TensorProto::DataType::UINT8) &&
      tensor_proto.int32_data().size() > 0) {
    out.reserve(tensor_proto.int32_data().size());
    for (int i = 0; i < tensor_proto.int32_data().size(); ++i) {
      out.push_back(static_cast<int64_t>(tensor_proto.int32_data()[i]));
    }
    return true;
  }
  if ((dtype == TensorProto::DataType::UINT64 || dtype == TensorProto::DataType::UINT32) &&
      tensor_proto.uint64_data().size() > 0) {
    out.reserve(tensor_proto.uint64_data().size());
    for (int i = 0; i < tensor_proto.uint64_data().size(); ++i) {
      out.push_back(static_cast<int64_t>(tensor_proto.uint64_data()[i]));
    }
    return true;
  }

  // Fall back to raw_data (little-endian fixed-width).
  if (!tensor_proto.is_raw_data()) {
    return false;
  }
  const utils::ByteSpan &raw = tensor_proto.raw_data();
  const uint8_t *bytes = raw.data();
  const size_t nbytes = raw.size();
  auto read_le = [&](size_t element_bytes, bool is_signed, size_t count) {
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      uint64_t u = 0;
      for (size_t b = 0; b < element_bytes; ++b) {
        u |= static_cast<uint64_t>(bytes[i * element_bytes + b]) << (8 * b);
      }
      int64_t v;
      if (is_signed) {
        // Sign-extend ``element_bytes``-wide value.
        const uint64_t sign_bit = uint64_t{1} << (element_bytes * 8 - 1);
        if (u & sign_bit) {
          // Fill the high bits with 1s.
          const uint64_t mask = ~((uint64_t{1} << (element_bytes * 8)) - 1);
          v = static_cast<int64_t>(u | mask);
        } else {
          v = static_cast<int64_t>(u);
        }
      } else {
        v = static_cast<int64_t>(u);
      }
      out.push_back(v);
    }
  };
  switch (dtype) {
  case TensorProto::DataType::INT64:
    if (nbytes % 8 != 0)
      return false;
    read_le(8, /*is_signed=*/true, nbytes / 8);
    return true;
  case TensorProto::DataType::UINT64:
    if (nbytes % 8 != 0)
      return false;
    read_le(8, /*is_signed=*/false, nbytes / 8);
    return true;
  case TensorProto::DataType::INT32:
    if (nbytes % 4 != 0)
      return false;
    read_le(4, /*is_signed=*/true, nbytes / 4);
    return true;
  case TensorProto::DataType::UINT32:
    if (nbytes % 4 != 0)
      return false;
    read_le(4, /*is_signed=*/false, nbytes / 4);
    return true;
  case TensorProto::DataType::INT16:
    if (nbytes % 2 != 0)
      return false;
    read_le(2, /*is_signed=*/true, nbytes / 2);
    return true;
  case TensorProto::DataType::UINT16:
    if (nbytes % 2 != 0)
      return false;
    read_le(2, /*is_signed=*/false, nbytes / 2);
    return true;
  case TensorProto::DataType::INT8:
    read_le(1, /*is_signed=*/true, nbytes);
    return true;
  case TensorProto::DataType::UINT8:
    read_le(1, /*is_signed=*/false, nbytes);
    return true;
  default:
    return false;
  }
}

bool ReadFloatingValues(const TensorProto &tensor_proto, std::vector<double> &out) {
  const auto dtype = tensor_proto.data_type();
  out.clear();

  // Type-specific storage takes precedence over raw_data when populated.
  if (dtype == TensorProto::DataType::FLOAT && tensor_proto.float_data().size() > 0) {
    out.reserve(tensor_proto.float_data().size());
    for (int i = 0; i < tensor_proto.float_data().size(); ++i) {
      out.push_back(static_cast<double>(tensor_proto.float_data()[i]));
    }
    return true;
  }
  if (dtype == TensorProto::DataType::DOUBLE && tensor_proto.double_data().size() > 0) {
    out.reserve(tensor_proto.double_data().size());
    for (int i = 0; i < tensor_proto.double_data().size(); ++i) {
      out.push_back(tensor_proto.double_data()[i]);
    }
    return true;
  }

  // Fall back to raw_data (little-endian fixed-width).
  if (!tensor_proto.is_raw_data()) {
    return false;
  }
  const utils::ByteSpan &raw = tensor_proto.raw_data();
  const uint8_t *bytes = raw.data();
  const size_t nbytes = raw.size();
  switch (dtype) {
  case TensorProto::DataType::FLOAT: {
    if (nbytes % 4 != 0) {
      return false;
    }
    const size_t count = nbytes / 4;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      uint32_t u = 0;
      for (size_t b = 0; b < 4; ++b) {
        u |= static_cast<uint32_t>(bytes[i * 4 + b]) << (8 * b);
      }
      float f;
      std::memcpy(&f, &u, sizeof(float));
      out.push_back(static_cast<double>(f));
    }
    return true;
  }
  case TensorProto::DataType::DOUBLE: {
    if (nbytes % 8 != 0) {
      return false;
    }
    const size_t count = nbytes / 8;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      uint64_t u = 0;
      for (size_t b = 0; b < 8; ++b) {
        u |= static_cast<uint64_t>(bytes[i * 8 + b]) << (8 * b);
      }
      double d;
      std::memcpy(&d, &u, sizeof(double));
      out.push_back(d);
    }
    return true;
  }
  default:
    return false;
  }
}

const GraphProto &FindGraphAttribute(const NodeProto &node, const char *attr_name,
                                     const char *context) {
  const std::string prefix =
      (context != nullptr && context[0] != '\0') ? (std::string(context) + ": ") : std::string();
  for (int i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    if (attr.name() != attr_name) {
      continue;
    }
    EXT_ENFORCE_INVALID(attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g(),
                        prefix + "attribute '" + attr_name +
                            "' must be a GRAPH on node of op_type '" + node.op_type().as_string() +
                            "'.");
    return attr.g();
  }
  throw std::invalid_argument(prefix + "attribute '" + attr_name +
                              "' is missing on node of op_type '" + node.op_type().as_string() +
                              "'.");
}

NodeProto MakeNode(const char *op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const char *domain, const char *name) {
  NodeProto node;
  node.set_op_type(std::string(op_type));
  for (const std::string &in : inputs) {
    node.add_input(in);
  }
  for (const std::string &out : outputs) {
    node.add_output(out);
  }
  if (domain != nullptr) {
    node.set_domain(std::string(domain));
  }
  if (name != nullptr) {
    node.set_name(std::string(name));
  }
  return node;
}

NodeProto &AddNode(GraphProto &graph, const char *op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const char *domain, const char *name) {
  NodeProto *node = graph.add_node();
  *node = MakeNode(op_type, inputs, outputs, domain, name);
  return *node;
}

namespace {

// Fills ``tensor`` with ``name`` and ``dims``; the caller is responsible for
// setting ``data_type`` and the typed payload field.
void InitInitializerHeader(TensorProto &tensor, const char *name,
                           const std::vector<int64_t> &dims) {
  tensor.set_name(std::string(name));
  for (int64_t d : dims) {
    tensor.add_dims(d);
  }
}

} // namespace

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<int64_t> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::INT64);
  for (int64_t v : values) {
    tensor.ref_int64_data().push_back(v);
  }
  return tensor;
}

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<int32_t> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::INT32);
  for (int32_t v : values) {
    tensor.ref_int32_data().push_back(v);
  }
  return tensor;
}

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<uint64_t> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::UINT64);
  for (uint64_t v : values) {
    tensor.ref_uint64_data().push_back(v);
  }
  return tensor;
}

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<float> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  for (float v : values) {
    tensor.ref_float_data().push_back(v);
  }
  return tensor;
}

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<double> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::DOUBLE);
  for (double v : values) {
    tensor.ref_double_data().push_back(v);
  }
  return tensor;
}

template <>
TensorProto MakeInitializer(const char *name, const std::vector<int64_t> &dims,
                            const std::vector<std::string> &values) {
  TensorProto tensor;
  InitInitializerHeader(tensor, name, dims);
  tensor.set_data_type(TensorProto::DataType::STRING);
  for (const std::string &v : values) {
    *tensor.add_string_data() = utils::String(v);
  }
  return tensor;
}

template <typename T>
TensorProto &AddInitializer(GraphProto &graph, const char *name, const std::vector<int64_t> &dims,
                            const std::vector<T> &values) {
  TensorProto *tensor = graph.add_initializer();
  *tensor = MakeInitializer<T>(name, dims, values);
  return *tensor;
}

// Explicit instantiations matching the MakeInitializer specializations above.
template TensorProto &AddInitializer<int64_t>(GraphProto &, const char *,
                                              const std::vector<int64_t> &,
                                              const std::vector<int64_t> &);
template TensorProto &AddInitializer<int32_t>(GraphProto &, const char *,
                                              const std::vector<int64_t> &,
                                              const std::vector<int32_t> &);
template TensorProto &AddInitializer<uint64_t>(GraphProto &, const char *,
                                               const std::vector<int64_t> &,
                                               const std::vector<uint64_t> &);
template TensorProto &AddInitializer<float>(GraphProto &, const char *,
                                            const std::vector<int64_t> &,
                                            const std::vector<float> &);
template TensorProto &AddInitializer<double>(GraphProto &, const char *,
                                             const std::vector<int64_t> &,
                                             const std::vector<double> &);
template TensorProto &AddInitializer<std::string>(GraphProto &, const char *,
                                                  const std::vector<int64_t> &,
                                                  const std::vector<std::string> &);

TensorProto MakeInitializerShape(const char *name, const std::vector<int64_t> &values) {
  return MakeInitializer<int64_t>(name, {static_cast<int64_t>(values.size())}, values);
}

TensorProto &AddInitializerShape(GraphProto &graph, const char *name,
                                 const std::vector<int64_t> &values) {
  return AddInitializer<int64_t>(graph, name, {static_cast<int64_t>(values.size())}, values);
}

} // namespace ONNX_LIGHT_NAMESPACE
