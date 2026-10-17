#include "stream.h"
#include "onnx_helper.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#if defined(SYS_openat2) && __has_include(<linux/openat2.h>)
#include <linux/openat2.h>
#define ONNX_LIGHT_HAS_OPENAT2
#endif
#endif
#else
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#define NOMINMAX
#include <windows.h>
#endif
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::utils {

namespace {

std::filesystem::path external_path_from_utf8(const std::string &value) {
#ifdef __cpp_char8_t
  return std::filesystem::path(reinterpret_cast<const char8_t *>(value.c_str()));
#else
  return std::filesystem::u8path(value);
#endif
}

std::string external_path_to_utf8(const std::filesystem::path &value) {
  const auto utf8 = value.u8string();
  return std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size());
}

#if defined(_WIN32) && defined(_MSC_VER)
void IgnoreInvalidParameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int,
                            uintptr_t) {}

class ScopedInvalidParameterHandler {
public:
  ScopedInvalidParameterHandler()
      : previous_(::_set_thread_local_invalid_parameter_handler(IgnoreInvalidParameter)) {}
  ~ScopedInvalidParameterHandler() { ::_set_thread_local_invalid_parameter_handler(previous_); }

private:
  _invalid_parameter_handler previous_;
};
#endif

std::filesystem::path normalized_model_parent(const std::string &model_path) {
  std::filesystem::path parent = std::filesystem::path(model_path).parent_path();
  if (parent.empty()) {
    return std::filesystem::path(".");
  }
  return parent.lexically_normal();
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
// Keeps the shared validation smaller than two inlined copies under the proto binary-size budget.
bool is_windows_compatible_filename(const std::filesystem::path &filename) {
  const auto &value = filename.native();
  return !value.empty() && value.back() != '.' && value.back() != ' ';
}

std::filesystem::path validate_external_location_is_next_to_model(const std::string &model_path,
                                                                  const std::string &location);

std::string validate_weights_file_is_next_to_model(const std::string &model_path,
                                                   const std::string &weights_file) {
  std::filesystem::path weights_path(weights_file);
  EXT_ENFORCE(!weights_path.empty(), "External weights file cannot be empty.");

  std::filesystem::path normalized_parent = normalized_model_parent(model_path);
  std::filesystem::path normalized_weights = weights_path.lexically_normal();
  auto relative = relative_external_data_path(normalized_weights, normalized_parent);
  if (!IsSafeExternalDataLocation(relative.string()) && weights_path.is_relative())
    relative = normalized_weights;
  return validate_external_location_is_next_to_model(model_path, relative.string()).string();
}

std::filesystem::path validate_external_location_is_next_to_model(const std::string &model_path,
                                                                  const std::string &location) {
  std::filesystem::path location_path(location);
  EXT_ENFORCE(
      IsSafeExternalDataLocation(location),
      "External data location must be a relative filename with at most one subdirectory. location=",
      location);

  std::filesystem::path normalized_location = location_path.lexically_normal();
  for (const auto &component : normalized_location) {
    if (component == std::filesystem::path("."))
      continue;
    EXT_ENFORCE(is_windows_compatible_filename(component),
                "External data location must include a filename that does not end with a dot or "
                "space. location=",
                location);
  }

  std::filesystem::path final_path = normalized_model_parent(model_path) / normalized_location;
  const auto base = std::filesystem::weakly_canonical(normalized_model_parent(model_path));
  const auto parent =
      std::filesystem::weakly_canonical(final_path.parent_path()).lexically_relative(base);
  EXT_ENFORCE(!parent.empty() && *parent.begin() != std::filesystem::path(".."),
              "External data location resolves outside model directory. location=", location);
  // Reject symlinks as write targets to prevent TOCTOU-based arbitrary file overwrites
  // (GHSA-8qff-7g33-75mx). Defense-in-depth: FileWriteStream also checks at open time.
  if (std::filesystem::is_symlink(final_path)) {
    EXT_THROW("External data location '", final_path.string(),
              "' is a symbolic link, which is not allowed for security reasons "
              "(see GHSA-8qff-7g33-75mx).");
  }
  return final_path;
}

std::string normalize_external_location(const std::string &location) {
  return location.empty() ? location : std::filesystem::path(location).lexically_normal().string();
}

std::shared_ptr<int> open_weights_file(const std::string &model_path,
                                       const std::string &weights_file, bool write) {
  const auto base = normalized_model_parent(model_path);
  const std::filesystem::path path =
      write
          ? std::filesystem::path(validate_weights_file_is_next_to_model(model_path, weights_file))
          : std::filesystem::path(weights_file);
  if (!write)
    validate_external_weights_read_path(path, base);
  const auto relative = relative_external_data_path(path, base);
  return OpenExternalDataFile(external_path_to_utf8(base), external_path_to_utf8(relative), write);
}

std::shared_ptr<int> MakeFileDescriptor() {
  return std::shared_ptr<int>(new int(-1), [](int *descriptor) {
    if (*descriptor >= 0) {
#ifdef _WIN32
      ::_close(*descriptor);
#else
      ::close(*descriptor);
#endif
    }
    delete descriptor;
  });
}

std::filesystem::path VerifyExternalContainment(const std::filesystem::path &path,
                                                const std::filesystem::path &base) {
  const auto canonical_base = std::filesystem::weakly_canonical(base);
  const auto canonical_path = std::filesystem::weakly_canonical(path);
  const auto relative = canonical_path.lexically_relative(canonical_base);
  EXT_ENFORCE(!relative.empty() && *relative.begin() != "..", "External data file '", path.string(),
              "' resolves outside the model directory '", base.string(), "'.");
  return canonical_path;
}

} // namespace

std::shared_ptr<int> OpenExternalDataFile(const std::string &base_dir,
                                          const std::string &relative_path, bool write) {
  EXT_ENFORCE(relative_path.find('\0') == std::string::npos &&
                  base_dir.find('\0') == std::string::npos,
              "External data paths must not contain null bytes.");
  const auto requested = external_path_from_utf8(relative_path);
  EXT_ENFORCE(!requested.empty() && !requested.has_root_path(),
              "External data location must be a relative path. location=", relative_path);
  for (const auto &component : requested)
    EXT_ENFORCE(component != "..", "External data location contains '..': ", relative_path);
  const auto relative = requested.lexically_normal();
  EXT_ENFORCE(relative != "." && !relative.filename().empty(),
              "External data location must include a filename. location=", relative_path);
  const auto base = external_path_from_utf8(base_dir.empty() ? "." : base_dir);
  const auto path = base / relative;
  VerifyExternalContainment(write ? path.parent_path() : path, base);
  EXT_ENFORCE(!std::filesystem::is_symlink(path),
              "External data is a symbolic link: ", relative_path);
  auto descriptor = MakeFileDescriptor();
#ifndef _WIN32
  const int flags = O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK | (write ? O_RDWR | O_CREAT : O_RDONLY);
  bool kernel_verified = false;
#if defined(ONNX_LIGHT_HAS_OPENAT2) || defined(O_RESOLVE_BENEATH)
  auto directory = MakeFileDescriptor();
  *directory = ::open(base.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (*directory >= 0) {
#ifdef ONNX_LIGHT_HAS_OPENAT2
    static thread_local bool openat2_supported = true;
    if (openat2_supported) {
      struct open_how how{};
      how.flags = static_cast<uint64_t>(flags);
      how.mode = write ? 0600 : 0;
      how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS;
      *descriptor =
          static_cast<int>(::syscall(SYS_openat2, *directory, relative.c_str(), &how, sizeof(how)));
      if (*descriptor < 0) {
        const int error = errno;
        EXT_ENFORCE(error == ENOSYS,
                    "Cannot open external data (kernel rejected path): ", relative_path,
                    " (errno=", error, ")");
        openat2_supported = false;
      }
    }
#else
    *descriptor = ::openat(*directory, relative.c_str(), flags | O_RESOLVE_BENEATH, 0600);
    EXT_ENFORCE(*descriptor >= 0,
                "Cannot open external data (kernel rejected path): ", relative_path,
                " (errno=", errno, ")");
#endif
    kernel_verified = *descriptor >= 0;
  }
#endif
  if (*descriptor < 0)
    *descriptor = ::open(path.c_str(), flags, 0600);
  EXT_ENFORCE(*descriptor >= 0, "Cannot open external data: ", relative_path, " (errno=", errno,
              ")");
  struct stat metadata{};
  EXT_ENFORCE(::fstat(*descriptor, &metadata) == 0,
              "Cannot query external data file information: ", relative_path);
  EXT_ENFORCE(S_ISREG(metadata.st_mode), "External data is not regular file: ", relative_path);
  EXT_ENFORCE(metadata.st_nlink == 1, "External data has multiple hard links: ", relative_path);
  if (!kernel_verified) {
    const auto canonical = VerifyExternalContainment(path, base);
    struct stat path_metadata{};
    EXT_ENFORCE(::stat(canonical.c_str(), &path_metadata) == 0,
                "Cannot query canonical external data file: ", relative_path);
    EXT_ENFORCE(metadata.st_dev == path_metadata.st_dev && metadata.st_ino == path_metadata.st_ino,
                "External data descriptor/path mismatch: ", relative_path);
  }
#else
  for (const auto &component : relative) {
    EXT_ENFORCE(component.native().find(L':') == std::wstring::npos,
                "External data components must not contain alternate data streams.");
  }
  HANDLE handle = ::CreateFileW(path.c_str(), write ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                write ? OPEN_ALWAYS : OPEN_EXISTING,
                                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  EXT_ENFORCE(handle != INVALID_HANDLE_VALUE, "Cannot open external data: ", relative_path);
  std::unique_ptr<void, decltype(&::CloseHandle)> file(handle, ::CloseHandle);
  BY_HANDLE_FILE_INFORMATION metadata{};
  EXT_ENFORCE(::GetFileInformationByHandle(handle, &metadata),
              "Cannot query external data file information.");
  EXT_ENFORCE(
      !(metadata.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) &&
          ::GetFileType(handle) == FILE_TYPE_DISK,
      "External data is a reparse point or not regular file.");
  EXT_ENFORCE(metadata.nNumberOfLinks == 1, "External data has multiple hard links.");
  const auto canonical = VerifyExternalContainment(path, base);
  HANDLE canonical_handle =
      ::CreateFileW(canonical.c_str(), FILE_READ_ATTRIBUTES,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  EXT_ENFORCE(canonical_handle != INVALID_HANDLE_VALUE,
              "Cannot open canonical external data file: ", relative_path);
  std::unique_ptr<void, decltype(&::CloseHandle)> canonical_file(canonical_handle, ::CloseHandle);
  BY_HANDLE_FILE_INFORMATION canonical_metadata{};
  EXT_ENFORCE(::GetFileInformationByHandle(canonical_handle, &canonical_metadata),
              "Cannot query canonical external data file information.");
  EXT_ENFORCE(metadata.dwVolumeSerialNumber == canonical_metadata.dwVolumeSerialNumber &&
                  metadata.nFileIndexHigh == canonical_metadata.nFileIndexHigh &&
                  metadata.nFileIndexLow == canonical_metadata.nFileIndexLow,
              "External data descriptor/path mismatch: ", relative_path);
  *descriptor = ::_open_osfhandle(reinterpret_cast<intptr_t>(handle),
                                  _O_BINARY | (write ? _O_RDWR : _O_RDONLY));
  EXT_ENFORCE(*descriptor >= 0, "Cannot convert external data handle to descriptor.");
  file.release();
#endif
  return descriptor;
}

std::filesystem::path relative_external_data_path(const std::filesystem::path &candidate_path,
                                                  const std::filesystem::path &base_dir) {
  const auto candidate = std::filesystem::absolute(candidate_path).lexically_normal();
  const auto base = std::filesystem::absolute(base_dir).lexically_normal();
  const auto relative = candidate.lexically_relative(base);
  if (!relative.empty() && *relative.begin() != "..")
    return relative;

  const auto canonical_base = std::filesystem::weakly_canonical(base);
  std::filesystem::path prefix;
  for (const auto &component : candidate.parent_path()) {
    prefix /= component;
    if (std::filesystem::weakly_canonical(prefix) == canonical_base)
      return candidate.lexically_relative(prefix);
  }
  return relative;
}

int64_t ExternalDataFileSize(int descriptor) {
#ifdef _WIN32
  LARGE_INTEGER size;
  EXT_ENFORCE(::GetFileSizeEx(reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor)), &size),
              "Cannot determine external data size.");
  return size.QuadPart;
#else
  struct stat metadata{};
  EXT_ENFORCE(::fstat(descriptor, &metadata) == 0, "Cannot determine external data size.");
  return metadata.st_size;
#endif
}

void ReadExternalDataFile(int descriptor, void *destination, int64_t size, int64_t offset) {
  EXT_ENFORCE(size >= 0 && offset >= 0 && size <= INT64_MAX - offset,
              "External data read is out of bounds.");
  EXT_ENFORCE(size == 0 || destination != nullptr, "External data read destination is null.");
  auto *bytes = static_cast<uint8_t *>(destination);
  int64_t done = 0;
  while (done < size) {
    const auto count = static_cast<unsigned int>(std::min<int64_t>(size - done, INT_MAX));
#ifdef _WIN32
    OVERLAPPED request{};
    const auto position = static_cast<uint64_t>(offset + done);
    request.Offset = static_cast<DWORD>(position);
    request.OffsetHigh = static_cast<DWORD>(position >> 32);
    DWORD result = 0;
    EXT_ENFORCE(::ReadFile(reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor)), bytes + done,
                           count, &result, &request),
                "External data read failed.");
#else
    const auto result = ::pread(descriptor, bytes + done, count, offset + done);
    if (result < 0 && errno == EINTR)
      continue;
#endif
    EXT_ENFORCE(result > 0, "External data read failed or short read.");
    done += result;
  }
}

void WriteExternalDataFile(int descriptor, const void *source, int64_t size, int64_t offset) {
  EXT_ENFORCE(size >= 0 && offset >= 0 && size <= INT64_MAX - offset,
              "External data write is out of bounds.");
  EXT_ENFORCE(size == 0 || source != nullptr, "External data write source is null.");
  const auto *bytes = static_cast<const uint8_t *>(source);
  int64_t done = 0;
  while (done < size) {
    const auto count = static_cast<unsigned int>(std::min<int64_t>(size - done, INT_MAX));
#ifdef _WIN32
    OVERLAPPED request{};
    const auto position = static_cast<uint64_t>(offset + done);
    request.Offset = static_cast<DWORD>(position);
    request.OffsetHigh = static_cast<DWORD>(position >> 32);
    DWORD result = 0;
    EXT_ENFORCE(::WriteFile(reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor)), bytes + done,
                            count, &result, &request),
                "External data write failed.");
#else
    const auto result = ::pwrite(descriptor, bytes + done, count, offset + done);
    if (result < 0 && errno == EINTR)
      continue;
#endif
    EXT_ENFORCE(result > 0, "External data write failed.");
    done += result;
  }
}

void TruncateExternalDataFile(int descriptor) {
#ifdef _WIN32
  HANDLE handle = reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor));
  LARGE_INTEGER zero{};
  EXT_ENFORCE(::SetFilePointerEx(handle, zero, nullptr, FILE_BEGIN),
              "External data truncate seek failed.");
  if (!::SetEndOfFile(handle))
    EXT_ENFORCE(::GetLastError() == ERROR_USER_MAPPED_FILE, "External data truncate failed.");
#else
  EXT_ENFORCE(::ftruncate(descriptor, 0) == 0, "External data truncate failed.");
#endif
}

std::shared_ptr<uint8_t> MapExternalDataFile(int descriptor) {
  const int64_t size = ExternalDataFileSize(descriptor);
  if (!size)
    return {};
  EXT_ENFORCE(static_cast<uint64_t>(size) <= SIZE_MAX, "External data mapping is too large.");
  const auto length = static_cast<size_t>(size);
#ifndef _WIN32
  void *mapped = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, descriptor, 0);
  EXT_ENFORCE(mapped != MAP_FAILED, "External data mmap failed.");
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [length](uint8_t *p) { ::munmap(p, length); });
#else
  HANDLE mapping = ::CreateFileMappingW(reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor)),
                                        nullptr, PAGE_READONLY, 0, 0, nullptr);
  EXT_ENFORCE(mapping != nullptr, "External data CreateFileMapping failed.");
  void *mapped = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, length);
  ::CloseHandle(mapping);
  EXT_ENFORCE(mapped != nullptr, "External data MapViewOfFile failed.");
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [](uint8_t *p) { ::UnmapViewOfFile(p); });
#endif
}

// Validates that *candidate_path* is not a symlink, resolves inside *base_dir*, and
// has no more than one hard link. Shared by external-data reads (TwoFilesStream) and
// the MappedPayload contract (onnx_mapped_payload.cc) so both apply the same confinement
// rules before a path is opened or memory-mapped.
void validate_external_weights_read_path(const std::filesystem::path &candidate_path,
                                         const std::filesystem::path &base_dir) {
  EXT_ENFORCE(!std::filesystem::is_symlink(candidate_path), "External data file '",
              candidate_path.string(), "' is a symbolic link, which is not allowed.");
  std::error_code ec;
  const std::filesystem::path canonical_data =
      std::filesystem::weakly_canonical(candidate_path, ec);
  EXT_ENFORCE(!ec, "External data path '", candidate_path.string(),
              "' could not be canonicalized: ", ec.message());
  const std::filesystem::path canonical_base = std::filesystem::weakly_canonical(base_dir, ec);
  EXT_ENFORCE(!ec, "Model base directory '", base_dir.string(),
              "' could not be canonicalized: ", ec.message());
  std::filesystem::path::string_type base_str = canonical_base.native();
  if (!base_str.empty() && base_str.back() != std::filesystem::path::preferred_separator) {
    base_str += std::filesystem::path::preferred_separator;
  }
  EXT_ENFORCE(canonical_data.native().find(base_str) == 0 || canonical_data == canonical_base,
              "External data file '", candidate_path.string(),
              "' resolves outside the model directory '", base_dir.string(), "'.");
  std::error_code ec_hc;
  const auto hc = std::filesystem::hard_link_count(candidate_path, ec_hc);
  EXT_ENFORCE(!ec_hc, "Could not determine hard link count for external data file '",
              candidate_path.string(), "': ", ec_hc.message());
  EXT_ENFORCE(hc <= 1, "External data file '", candidate_path.string(),
              "' has multiple hard links (", static_cast<int64_t>(hc), "), which is not allowed.");
}

// Maps an entire file into read-only virtual memory and returns a shared_ptr<uint8_t>
// whose deleter unmaps the region.  On POSIX, mmap(MAP_PRIVATE|PROT_READ) is used;
// on Windows, CreateFileMapping + MapViewOfFile.
// Returns an empty shared_ptr when file_size == 0.
// The mapped base address is always page-aligned, which satisfies any typical tensor
// alignment requirement (16 / 32 / 64 bytes) when combined with an aligned file offset.
std::shared_ptr<uint8_t> mmap_file_as_shared_ptr(const std::string &file_path, int64_t file_size) {
  if (file_size <= 0) {
    return std::shared_ptr<uint8_t>();
  }
  const size_t sz = static_cast<size_t>(file_size);
#if !defined(_WIN32)
  const int fd = ::open(file_path.c_str(), O_RDONLY);
  EXT_ENFORCE(fd >= 0, "mmap_file: Unable to open file: ", file_path);
  void *mapped = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  EXT_ENFORCE(mapped != MAP_FAILED, "mmap_file: mmap failed for file: ", file_path);
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [sz](uint8_t *ptr) { ::munmap(ptr, sz); });
#else
  HANDLE hFile = ::CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  EXT_ENFORCE(hFile != INVALID_HANDLE_VALUE, "mmap_file: Unable to open file: ", file_path);
  HANDLE hMapping = ::CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  ::CloseHandle(hFile);
  EXT_ENFORCE(hMapping != nullptr, "mmap_file: CreateFileMapping failed for file: ", file_path);
  void *mapped = ::MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
  ::CloseHandle(hMapping);
  EXT_ENFORCE(mapped != nullptr, "mmap_file: MapViewOfFile failed for file: ", file_path);
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [](uint8_t *ptr) { ::UnmapViewOfFile(ptr); });
#endif
}

namespace {

#if !defined(_WIN32)

// Reads a delayed block from a shared file descriptor using positional reads.
// It retries on EINTR and enforces full reads to avoid truncated tensor payloads.
void ReadBlockFromFd(int fd, const DelayedBlock &block, const char *context) {
  size_t done = 0;
  while (done < block.size) {
    ssize_t bytes_read =
        pread(fd, block.data + done, block.size - done, static_cast<off_t>(block.offset + done));
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int err = errno;
      EXT_THROW(context, " failed to read delayed block at offset=", block.offset, ", errno=", err,
                " (", strerror(err), ")");
    }
    EXT_ENFORCE(bytes_read > 0, context,
                " reached end of file while reading delayed block at offset=", block.offset,
                ", expected=", block.size, ", read=", done);
    done += static_cast<size_t>(bytes_read);
  }
}

#endif

} // namespace

///////////////
// BinaryStream
///////////////

std::string FieldNumber::string() const {
  return onnx_light_helpers::MakeString("[field_number=", field_number, ", wire_type=", wire_type,
                                        "]");
}

void BinaryStream::_check() {
  EXT_ENFORCE(limits_.empty(), "BinaryStream destructor called with non-empty limits stack.");
}

BinaryStream::~BinaryStream() { _check(); }

RefString BinaryStream::next_string() {
  // Depending on the stream implementation, the string may be disappear after reading another item.
  uint64_t length = next_uint64();
  this->CanRead(length, "[BinaryStream::next_string]");
  return RefString(reinterpret_cast<const char *>(read_bytes(length)), static_cast<size_t>(length));
}

int64_t BinaryStream::next_int64() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int64_t>(value);
}

int32_t BinaryStream::next_int32() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int32_t>(value);
}

float BinaryStream::next_float() {
  float value;
  read_bytes(sizeof(float), reinterpret_cast<uint8_t *>(&value));
  return value;
}

double BinaryStream::next_double() {
  double value;
  read_bytes(sizeof(double), reinterpret_cast<uint8_t *>(&value));
  return value;
}

FieldNumber BinaryStream::next_field() {
  FieldNumber n;
  n.wire_type = next_uint64();
  n.field_number = n.wire_type >> 3;
  n.wire_type = n.wire_type & 0x07;
  return n;
}

void BinaryStream::ReadDelayedBlock(DelayedBlock &) {
  EXT_THROW("ReadDelayedBlock is not implemented for this stream.");
}

void BinaryStream::WaitForDelayedBlock() {
  EXT_THROW("WaitForDelayedBlock is not implemented for this stream.");
}

void BinaryStream::StartThreadPool(size_t) {
  EXT_THROW("StartThreadPool is not implemented for this stream.");
}

void BinaryStream::LimitToNext(uint64_t length) {
  CanRead(length, "Too many bytes requested in LimitToNext.");
  limits_.push_back(size());
  LimitTo(tell() + length);
}

void BinaryStream::Restore() {
  EXT_ENFORCE(!limits_.empty(), "Cannot restore, no limits set");
  uint64_t last_limit = limits_.back();
  LimitTo(last_limit);
  limits_.pop_back();
}

bool BinaryStream::Next(const void **data, int *size) {
  EXT_ENFORCE(data != nullptr && size != nullptr,
              "BinaryStream::Next: output pointers must not be null.");
  if (backed_up_ > 0) {
    // Re-serve the tail of the previously returned block without touching the
    // underlying stream (the position was already advanced past it).
    const uint8_t *ptr = last_next_data_ + (last_next_size_ - backed_up_);
    *data = ptr;
    *size = static_cast<int>(backed_up_);
    last_next_data_ = ptr;
    last_next_size_ = backed_up_;
    backed_up_ = 0;
    return true;
  }
  if (!NotEnd())
    return false;
  EXT_ENFORCE(CanNoCopy(), "BinaryStream::Next requires a no-copy (in-memory) stream so a "
                           "pointer into the backing buffer can be returned.");
  // ZeroCopyInputStream blocks are int-sized; cap huge remainders so the next
  // call keeps serving the rest.
  constexpr offset_t kMaxChunk = 0x7FFFFFFF;
  offset_t remaining = this->size() - tell();
  if (remaining > kMaxChunk)
    remaining = kMaxChunk;
  const uint8_t *ptr = read_bytes(remaining);
  *data = ptr;
  *size = static_cast<int>(remaining);
  last_next_data_ = ptr;
  last_next_size_ = remaining;
  return true;
}

void BinaryStream::BackUp(int count) {
  EXT_ENFORCE(count >= 0, "BinaryStream::BackUp: count must be non-negative, got ", count, ".");
  EXT_ENFORCE(static_cast<int64_t>(count) <= last_next_size_, "BinaryStream::BackUp: count (",
              count, ") exceeds the size of the last Next() block (", last_next_size_, ").");
  backed_up_ = count;
}

int64_t BinaryStream::ByteCount() const { return tell() - backed_up_; }

///////////////
// StringStream
///////////////

void StringStream::Setup(const uint8_t *data, int64_t size) {
  EXT_ENFORCE(!thread_pool_.IsStarted(), "ThreadPool is still running.");
  pos_ = 0;
  size_ = size;
  data_ = data;
  thread_pool_.Clear();
  blocks_.clear();
}

void StringStream::CanRead(uint64_t len, const char *msg) {
  // Use unsigned arithmetic to avoid signed overflow / signed-cast bypass when
  // ``len`` comes from an untrusted varint and is > INT64_MAX. ``pos_`` and
  // ``size_`` are non-negative by construction (pos_ is advanced from 0 and
  // never past size_), so the subtraction below cannot underflow.
  EXT_ENFORCE(pos_ >= 0 && size_ >= 0 && pos_ <= size_ &&
                  static_cast<uint64_t>(size_ - pos_) >= len,
              msg, " unable to read ", len, " bytes, pos_=", pos_, ", size_=", size_);
}

const uint8_t *StringStream::read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) {
  EXT_ENFORCE(n_bytes >= 0 && pos_ + n_bytes <= size_,
              "[StringStream::read_bytes] out of bounds: pos=", pos_, ", n_bytes=", n_bytes,
              ", size=", size_);
  if (pre_allocated_buffer != nullptr) {
    memcpy(pre_allocated_buffer, data_ + pos_, n_bytes);
    pos_ += n_bytes;
    return pre_allocated_buffer;
  } else {
    const uint8_t *res = data_ + pos_;
    pos_ += n_bytes;
    return res;
  }
}

void StringStream::skip_bytes(offset_t n_bytes) {
  EXT_ENFORCE(n_bytes >= 0 && pos_ + n_bytes <= size_,
              "[StringStream::skip_bytes] out of bounds: pos=", pos_, ", n_bytes=", n_bytes,
              ", size=", size_);
  pos_ += n_bytes;
}

uint64_t StringStream::next_uint64() {
  // Fast path: single-byte varint (covers field numbers 1-15 and other small values).
  if (pos_ < size_ && (data_[pos_] & 0x80) == 0) {
    return static_cast<uint64_t>(data_[pos_++]);
  }

  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10 && pos_ < size_; ++i) {
    uint8_t byte = data_[pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[StringStream::next_uint64] unable to read an uint64 at pos=", pos_, ", size=", size_);
}

std::string StringStream::tell_around() const {
  offset_t begin = pos_;
  offset_t end =
      pos_ + 10 < static_cast<offset_t>(size()) ? pos_ + 10 : static_cast<offset_t>(size());
  RefString ref(reinterpret_cast<const char *>(data_) + begin, end - begin);
  return std::string(ref);
}

void StringStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void StringStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(block.stream_id == 0,
              "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask(
      [this, block]() { memcpy(block.data, this->data_ + block.offset, block.size); });
  pos_ += block.size;
}

void StringStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void StringStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(static_cast<int32_t>(n_threads));
}

////////////////////
// BinaryWriteStream
////////////////////

void BinaryWriteStream::write_variant_uint64(uint64_t value) {
  // A 64-bit varint encodes at most 10 bytes (ceil(64/7)).
  uint8_t buf[10];
  int len = 0;
  while (value > 127) {
    buf[len++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
    value >>= 7;
  }
  buf[len++] = static_cast<uint8_t>(value);
  write_raw_bytes(buf, len);
}

uint64_t BinaryWriteStream::size_variant_uint64(uint64_t value) { return VarintSize(value); }

void BinaryWriteStream::write_int64(int64_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int64(int64_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_int32(int32_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int32(int32_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_float(float value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(float));
}

uint64_t BinaryWriteStream::size_float(float) { return sizeof(float); }

void BinaryWriteStream::write_double(double value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(double));
}

uint64_t BinaryWriteStream::size_double(double) { return sizeof(double); }

void BinaryWriteStream::write_field_header(uint32_t field_number, uint8_t wire_type) {
  write_variant_uint64((field_number << 3) | wire_type);
}

uint64_t BinaryWriteStream::VarintSize(uint64_t value) {
  size_t size = 0;
  do {
    size++;
    value >>= 7;
  } while (value != 0);
  return size;
}

uint64_t BinaryWriteStream::size_field_header(uint32_t field_number, uint8_t wire_type) {
  return VarintSize((field_number << 3) | wire_type);
}

void BinaryWriteStream::write_string(const std::string &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const std::string &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const RefString &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const RefString &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string_stream(const StringWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const StringWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::write_string_stream(const BorrowedWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const BorrowedWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::CacheSize(const void *ptr, uint16_t proto_id, SerializeSizeResult size) {
  size_cache_[{ptr, proto_id}] = size;
}

bool BinaryWriteStream::GetCachedSize(const void *ptr, uint16_t proto_id,
                                      SerializeSizeResult &size) {
  auto it = size_cache_.find({ptr, proto_id});
  if (it != size_cache_.end()) {
    size = it->second;
    return true;
  }
  return false;
}

void BinaryWriteStream::StartThreadPool(size_t) {}

void BinaryWriteStream::WriteDelayedBlock(DelayedWriteBlock &) {
  EXT_THROW("WriteDelayedBlock is not implemented for this stream.");
}

void BinaryWriteStream::WaitForDelayedBlock() {}

////////////////////
// StringWriteStream
////////////////////

// Note: The serialization thread exclusively calls write_raw_bytes,
// WriteDelayedBlock, and all write_pos_ updates.  Worker threads only
// execute the memcpy inside the submitted tasks and never touch write_pos_
// or resize the buffer.

void StringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  if (write_pos_ + n_bytes > static_cast<offset_t>(buffer_.size())) {
    // Geometric growth to amortize repeated small writes.
    size_t required = static_cast<size_t>(write_pos_ + n_bytes);
    size_t new_capacity = buffer_.size() + buffer_.size() / 2;
    if (new_capacity < required)
      new_capacity = required;
    buffer_.resize(new_capacity);
  }
  std::memcpy(buffer_.data() + write_pos_, ptr, static_cast<size_t>(n_bytes));
  write_pos_ += n_bytes;
}

int64_t StringWriteStream::size() const { return write_pos_; }
const uint8_t *StringWriteStream::data() const { return buffer_.data(); }

void StringWriteStream::pre_allocate(int64_t total_bytes) {
  buffer_.assign(static_cast<size_t>(total_bytes), 0);
  write_pos_ = 0;
}

bool StringWriteStream::Next(void **data, int *size) {
  EXT_ENFORCE(data != nullptr && size != nullptr,
              "StringWriteStream::Next: output pointers must not be null.");
  // Ensure there is spare room past the current write position, growing the
  // buffer geometrically (matching write_raw_bytes' amortized growth).
  if (write_pos_ >= static_cast<offset_t>(buffer_.size())) {
    size_t old_capacity = buffer_.size();
    size_t new_capacity = old_capacity + old_capacity / 2;
    constexpr size_t kMinBlock = 1024;
    if (new_capacity < old_capacity + kMinBlock)
      new_capacity = old_capacity + kMinBlock;
    buffer_.resize(new_capacity);
  }
  // ZeroCopyOutputStream blocks are int-sized; cap the handed-out span.
  constexpr offset_t kMaxChunk = 0x7FFFFFFF;
  offset_t available = static_cast<offset_t>(buffer_.size()) - write_pos_;
  if (available > kMaxChunk)
    available = kMaxChunk;
  *data = buffer_.data() + write_pos_;
  *size = static_cast<int>(available);
  write_pos_ += available;
  return true;
}

void StringWriteStream::BackUp(int count) {
  EXT_ENFORCE(count >= 0, "StringWriteStream::BackUp: count must be non-negative, got ", count,
              ".");
  EXT_ENFORCE(static_cast<offset_t>(count) <= write_pos_, "StringWriteStream::BackUp: count (",
              count, ") exceeds bytes produced (", write_pos_, ").");
  write_pos_ -= count;
}

void StringWriteStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(static_cast<int32_t>(n_threads));
}

void StringWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(write_pos_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(write_pos_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and write_pos_=", write_pos_);
  // The buffer must have been pre-allocated (via pre_allocate) before parallel
  // writes begin so that buffer_.data() remains stable during concurrent tasks.
  EXT_ENFORCE(block.offset + static_cast<int64_t>(block.size) <=
                  static_cast<int64_t>(buffer_.size()),
              "Buffer not pre-allocated: delayed write at offset=", block.offset,
              " size=", block.size, " exceeds buffer size=", buffer_.size());
  write_pos_ += static_cast<offset_t>(block.size);
  uint8_t *dest = reinterpret_cast<uint8_t *>(buffer_.data()) + block.offset;
  // block.data must remain valid (point to live proto data) until WaitForDelayedBlock() returns.
  // This is the caller's responsibility, identical to the FileWriteStream contract.
  thread_pool_.SubmitTask([dest, block]() { std::memcpy(dest, block.data, block.size); });
}

void StringWriteStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

//////////////////////////
// BorrowedStringWriteStream
//////////////////////////

void BorrowedStringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  EXT_ENFORCE(write_pos_ + n_bytes <= capacity_, "Buffer too small: write at offset=", write_pos_,
              " size=", n_bytes, " exceeds capacity=", capacity_);
  std::memcpy(data_ + write_pos_, ptr, static_cast<size_t>(n_bytes));
  write_pos_ += n_bytes;
}

bool BorrowedStringWriteStream::Next(void **data, int *size) {
  EXT_ENFORCE(data != nullptr && size != nullptr,
              "BorrowedStringWriteStream::Next: output pointers must not be null.");
  // Fixed-capacity buffer: never grows. Report exhaustion by returning false.
  if (write_pos_ >= capacity_)
    return false;
  constexpr offset_t kMaxChunk = 0x7FFFFFFF;
  offset_t available = capacity_ - write_pos_;
  if (available > kMaxChunk)
    available = kMaxChunk;
  *data = data_ + write_pos_;
  *size = static_cast<int>(available);
  write_pos_ += available;
  return true;
}

void BorrowedStringWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(write_pos_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(write_pos_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and write_pos_=", write_pos_);
  EXT_ENFORCE(block.offset + static_cast<int64_t>(block.size) <= capacity_,
              "Buffer too small: delayed write at offset=", block.offset, " size=", block.size,
              " exceeds capacity=", capacity_);
  write_pos_ += static_cast<offset_t>(block.size);
  uint8_t *dest = data_ + block.offset;
  thread_pool_.SubmitTask([dest, block]() { std::memcpy(dest, block.data, block.size); });
}

////////////////
// FdWriteStream
////////////////

namespace {

// Writes the full buffer to a raw file descriptor, looping over partial writes.
void fd_write_all(int fd, const char *data, size_t n_bytes) {
  size_t written = 0;
  while (written < n_bytes) {
    size_t remaining = n_bytes - written;
#if defined(_WIN32)
    unsigned int chunk =
        remaining > 0x7FFFFFFFu ? 0x7FFFFFFFu : static_cast<unsigned int>(remaining);
    int n = _write(fd, data + written, chunk);
#else
    size_t chunk = remaining;
    ssize_t n = ::write(fd, data + written, chunk);
#endif
    EXT_ENFORCE(n > 0, "FdWriteStream: failed to write ", static_cast<int64_t>(remaining),
                " bytes to file descriptor ", fd, " (errno=", errno, ").");
    written += static_cast<size_t>(n);
  }
}

} // namespace

void FdWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  Flush();
  fd_write_all(fd_, reinterpret_cast<const char *>(data), static_cast<size_t>(n_bytes));
  written_ += static_cast<int64_t>(n_bytes);
}

bool FdWriteStream::Flush() {
  if (used_ > 0) {
    fd_write_all(fd_, buffer_, used_);
    written_ += static_cast<int64_t>(used_);
    used_ = 0;
  }
  return true;
}

////////////////////
// FdReadStream
////////////////////

int64_t FdReadStream::InitLimit(int fd) noexcept {
#if !defined(_WIN32)
  auto cur = ::lseek(fd, 0, SEEK_CUR);
  if (cur < 0)
    return INT64_MAX; // non-seekable fd (pipe, socket, etc.)
  auto end = ::lseek(fd, 0, SEEK_END);
  if (end < 0) {
    ::lseek(fd, cur, SEEK_SET);
    return INT64_MAX;
  }
  ::lseek(fd, cur, SEEK_SET);
  return end >= cur ? static_cast<int64_t>(end - cur) : 0; // remaining bytes
#else
  // Avoid probing descriptor positions on Windows. Some CRT calls can trigger
  // fail-fast invalid-parameter handling for invalid/closed descriptors, while
  // ParseFromFileDescriptor read-failure paths are expected to return false.
  (void)fd;
  return INT64_MAX;
#endif
}

int FdReadStream::AdaptiveBlockSize(int64_t limit) noexcept {
  if (limit == INT64_MAX)
    return 64 * 1024;
  if (limit <= 64 * 1024)
    return 4096;
  if (limit <= 16 * 1024 * 1024)
    return 64 * 1024;
  return 1024 * 1024;
}

FdReadStream::FdReadStream(int fd, int block_size)
    : BinaryStream(), fd_(fd), block_size_(block_size), buffer_(nullptr), available_(0),
      position_(0), last_returned_(0), total_read_(0), pos_(0), limit_(InitLimit(fd_)), eof_(false),
      failed_(false) {
  if (block_size_ <= 0)
    block_size_ = AdaptiveBlockSize(limit_);
  buffer_ = new char[static_cast<size_t>(block_size_)];
}

FdReadStream::~FdReadStream() { delete[] buffer_; }

bool FdReadStream::EnsureAvailable() const {
  if (available_ > 0)
    return true;
  position_ = 0;
  int64_t n;
#if !defined(_WIN32)
  do {
    n = ::read(fd_, buffer_, static_cast<size_t>(block_size_));
  } while (n < 0 && errno == EINTR);
#else
#if defined(_MSC_VER)
  intptr_t handle = -1;
  {
    ScopedInvalidParameterHandler guard;
    handle = ::_get_osfhandle(fd_);
  }
#else
  const intptr_t handle = ::_get_osfhandle(fd_);
#endif
  if (handle == -1) {
    errno = EBADF;
    n = -1;
  } else {
    DWORD bytes_read = 0;
    const BOOL ok = ::ReadFile(reinterpret_cast<HANDLE>(handle), buffer_,
                               static_cast<DWORD>(block_size_), &bytes_read, nullptr);
    if (!ok) {
      const DWORD err = ::GetLastError();
      if (err == ERROR_BROKEN_PIPE || err == ERROR_HANDLE_EOF) {
        n = 0;
      } else {
        errno = (err == ERROR_INVALID_HANDLE) ? EBADF : EIO;
        n = -1;
      }
    } else {
      n = static_cast<int64_t>(bytes_read);
    }
  }
#endif
  if (n <= 0) {
    failed_ = n < 0;
    eof_ = true;
    return false;
  }
  available_ = static_cast<int>(n);
  total_read_ += static_cast<int64_t>(n);
  return true;
}

bool FdReadStream::NotEnd() const {
  if (pos_ >= limit_)
    return false;
  return EnsureAvailable();
}

bool FdReadStream::Next(const void **data, int *size) {
  EXT_ENFORCE(data != nullptr && size != nullptr,
              "FdReadStream::Next: output pointers must not be null.");
  if (!EnsureAvailable()) {
    *data = nullptr;
    *size = 0;
    return false;
  }
  *data = buffer_ + position_;
  *size = available_;
  last_returned_ = available_;
  position_ += available_;
  available_ = 0;
  return true;
}

void FdReadStream::BackUp(int count) {
  EXT_ENFORCE(count >= 0 && count <= last_returned_, "FdReadStream::BackUp: invalid count ", count,
              ".");
  position_ -= count;
  available_ += count;
  last_returned_ = 0;
}

bool FdReadStream::Skip(int count) {
  if (count < 0 || pos_ < 0 || (limit_ != INT64_MAX && limit_ - pos_ < count))
    return false;
  int64_t remaining = count;
  while (remaining > 0) {
    if (!EnsureAvailable())
      return false;
    int consumed = static_cast<int>(std::min<int64_t>(remaining, available_));
    position_ += consumed;
    available_ -= consumed;
    pos_ += consumed;
    remaining -= consumed;
  }
  return true;
}

uint64_t FdReadStream::next_uint64() {
  uint64_t result = 0;
  int shift = 0;
  for (int i = 0; i < 10; ++i) {
    uint8_t byte = 0;
    read_bytes(1, &byte);
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0)
      return result;
    shift += 7;
  }
  EXT_THROW("[FdReadStream::next_uint64] invalid varint at pos=", pos_);
}

void FdReadStream::CanRead(uint64_t len, const char *msg) {
  auto remaining = limit_ - pos_;
  EXT_ENFORCE(remaining >= 0 && static_cast<uint64_t>(remaining) >= len, msg, " unable to read ",
              len, " bytes; pos=", pos_, ", limit=", limit_);
}

std::string FdReadStream::tell_around() const {
  return std::string("[fd stream at pos=") + std::to_string(pos_) + "]";
}

const uint8_t *FdReadStream::read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) {
  EXT_ENFORCE(n_bytes >= 0, "[FdReadStream::read_bytes] n_bytes must be non-negative.");
  CanRead(static_cast<uint64_t>(n_bytes), "[FdReadStream::read_bytes]");
  // When no destination is supplied, allocate and own a small buffer for this read
  // (rather than throwing): file-descriptor-backed streams cannot hand out a stable
  // pointer into their own rotating block_size_ buffer (each Next() call overwrites
  // it), but callers of the zero-copy read path (BinaryStream::next_string() for
  // RefString/String fields, next_packed_element<T>() for scalar fixed-width fields)
  // only need the returned pointer to remain valid for the lifetime of this stream.
  std::unique_ptr<uint8_t[]> owned;
  if (pre_allocated_buffer == nullptr) {
    owned = std::make_unique<uint8_t[]>(static_cast<size_t>(n_bytes));
    pre_allocated_buffer = owned.get();
  }
  auto *dst = pre_allocated_buffer;
  offset_t remaining = n_bytes;
  while (remaining > 0) {
    const void *data = nullptr;
    int avail = 0;
    if (!Next(&data, &avail))
      EXT_THROW("[FdReadStream::read_bytes] EOF reached with ", remaining,
                " bytes still needed at pos=", pos_);
    auto to_copy = static_cast<int64_t>(avail) < remaining ? avail : static_cast<int>(remaining);
    std::memcpy(dst, data, static_cast<size_t>(to_copy));
    dst += to_copy;
    remaining -= static_cast<offset_t>(to_copy);
    if (to_copy < avail)
      BackUp(avail - to_copy);
  }
  pos_ += n_bytes;
  if (owned)
    owned_blocks_.push_back(std::move(owned));
  return pre_allocated_buffer;
}

void FdReadStream::skip_bytes(offset_t n_bytes) {
  EXT_ENFORCE(n_bytes >= 0, "[FdReadStream::skip_bytes] n_bytes must be non-negative.");
  CanRead(static_cast<uint64_t>(n_bytes), "[FdReadStream::skip_bytes]");
  while (n_bytes > 0) {
    EXT_ENFORCE(EnsureAvailable(), "[FdReadStream::skip_bytes] EOF reached with ", n_bytes,
                " bytes still needed at pos=", pos_);
    int consumed = static_cast<int>(std::min<int64_t>(n_bytes, available_));
    position_ += consumed;
    available_ -= consumed;
    pos_ += consumed;
    n_bytes -= consumed;
  }
}

//////////////////////
// BorrowedWriteStream
//////////////////////

void BorrowedWriteStream::write_raw_bytes(const uint8_t *, offset_t) {
  EXT_THROW("This method cannot be called on this class (BorrowedWriteStream).");
}

//////////////////
// FileWriteStream
//////////////////

FileWriteStream::FileWriteStream(const std::string &file_path)
    : BinaryWriteStream(), file_path_(file_path) {
  written_bytes_ = 0;
  // Normal model output retains its historical pathname guard. External-data
  // writers use the retained-file overload below instead.
  if (std::filesystem::is_symlink(std::filesystem::path(file_path))) {
    EXT_THROW("FileWriteStream: target path '", file_path,
              "' is a symbolic link, which is not allowed for security reasons "
              "(see GHSA-8qff-7g33-75mx).");
  }
  file_stream_.open(file_path, std::ios::binary);
  if (!file_stream_.is_open() && std::filesystem::exists(file_path)) {
    // On Windows, opening an existing file with truncation (the mode above) fails with
    // ERROR_USER_MAPPED_FILE when some other memory-mapped view of the same path is still
    // alive — e.g. re-saving a no_copy-loaded model back to its own external-data location,
    // whose bytes are still borrowed from an active MapViewOfFile mapping. Retry without
    // requesting truncation: this only avoids resizing the file down to zero at open time;
    // pre_allocate()/write_raw_bytes_at_offset() below can still seek past the current end to
    // grow the file as needed.
    file_stream_.clear();
    file_stream_.open(file_path, std::ios::binary | std::ios::in | std::ios::out);
  }
  EXT_ENFORCE(file_stream_.is_open(),
              "FileWriteStream: failed to open file for writing: ", file_path);
}

FileWriteStream::FileWriteStream(const std::string &file_path, std::shared_ptr<int> file)
    : file_path_(file_path), external_file_(std::move(file)), written_bytes_(0) {
  TruncateExternalDataFile(*external_file_);
}

void FileWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  if (external_file_)
    WriteExternalDataFile(*external_file_, data, n_bytes, static_cast<int64_t>(written_bytes_));
  else
    file_stream_.write(reinterpret_cast<const char *>(data), n_bytes);
  written_bytes_ += static_cast<uint64_t>(n_bytes);
}

int64_t FileWriteStream::size() const { return static_cast<int64_t>(written_bytes_); }

const uint8_t *FileWriteStream::data() const {
  EXT_THROW("This method cannot be called on this class (FileWriteStream).");
}

void FileWriteStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(static_cast<int32_t>(n_threads));
}

void FileWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0,
      "Only one stream is allowed to write delayed blocks, but stream_id=", block.stream_id);
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(written_bytes_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(written_bytes_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and written_bytes_=", written_bytes_);
  if (!external_file_)
    file_stream_.seekp(static_cast<std::streamoff>(block.size), std::ios::cur);
  written_bytes_ += static_cast<uint64_t>(block.size);
  if (external_file_) {
    auto file = external_file_;
    thread_pool_.SubmitTask(
        [file, block]() { WriteExternalDataFile(*file, block.data, block.size, block.offset); });
    return;
  }
  std::string file_path = file_path_;
  thread_pool_.SubmitTask([file_path, block]() {
    std::fstream file_stream(file_path, std::ios::in | std::ios::out | std::ios::binary);
    file_stream.seekp(block.offset);
    file_stream.write(reinterpret_cast<const char *>(block.data), block.size);
  });
}

void FileWriteStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void FileWriteStream::pre_allocate(int64_t total_bytes) {
  EXT_ENFORCE(total_bytes > 0, "pre_allocate requires total_bytes > 0, got ", total_bytes);
  if (external_file_) {
    const uint8_t zero = 0;
    WriteExternalDataFile(*external_file_, &zero, 1, total_bytes - 1);
    written_bytes_ = static_cast<uint64_t>(total_bytes);
    return;
  }
  // Seek to the last byte and write a zero to establish the file size.
  // Flush immediately so the ofstream buffer is empty; otherwise the destructor
  // would re-flush this sentinel byte and overwrite the last byte written by a
  // parallel task that fills the file concurrently.
  file_stream_.seekp(total_bytes - 1);
  const uint8_t zero = 0;
  file_stream_.write(reinterpret_cast<const char *>(&zero), 1);
  file_stream_.flush();
  written_bytes_ = static_cast<uint64_t>(total_bytes);
}

void FileWriteStream::write_raw_bytes_at_offset(const uint8_t *data, offset_t n_bytes,
                                                int64_t offset) {
  if (external_file_) {
    WriteExternalDataFile(*external_file_, data, n_bytes, offset);
    return;
  }
  file_stream_.seekp(offset);
  file_stream_.write(reinterpret_cast<const char *>(data), n_bytes);
  EXT_ENFORCE(!file_stream_.fail(), "Write failed for file: ", file_path_, " at offset=", offset,
              " n_bytes=", n_bytes);
}

/////////////
// FileStream
/////////////

FileStream::FileStream(const std::string &file_path)
    : lock_(false), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  if (!file_stream_.is_open()) {
    EXT_THROW("Unable to open file: ", file_path);
  }
#if !defined(_WIN32)
  file_descriptor_ = open(file_path.c_str(), O_RDONLY);
  if (file_descriptor_ < 0) {
    const int err = errno;
    EXT_THROW("Unable to open file descriptor for: ", file_path, ", errno=", err, " (",
              strerror(err), ")");
  }
#endif
  file_stream_.seekg(0, std::ios::end);
  std::streampos end = file_stream_.tellg();
  EXT_ENFORCE(end != std::streampos(-1) && static_cast<std::streamoff>(end) >= 0,
              "Unable to determine size of file (tellg failed): ", file_path);
  file_stream_.seekg(0);
  size_ = static_cast<offset_t>(end);
  read_buf_.resize(READ_BUF_SIZE);
}

FileStream::FileStream(const std::string &file_path, std::shared_ptr<int> file)
    : lock_(false), file_path_(file_path), external_file_(std::move(file)),
      size_(ExternalDataFileSize(*external_file_)) {
  read_buf_.resize(READ_BUF_SIZE);
}

bool FileStream::is_open() const { return external_file_ || file_stream_.is_open(); }

void FileStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void FileStream::CanRead(uint64_t len, const char *msg) {
  // Use unsigned arithmetic to avoid signed overflow / signed-cast bypass when
  // ``len`` comes from an untrusted varint and is > INT64_MAX. ``tell()`` and
  // ``size_`` are non-negative by construction.
  const int64_t cur = tell();
  EXT_ENFORCE(cur >= 0 && size_ >= 0 && cur <= size_ && static_cast<uint64_t>(size_ - cur) >= len,
              msg, " unable to read ", len, " bytes, pos_=", cur, ", size_=", size_);
}

void FileStream::_fill_read_buffer() {
  int64_t avail = size_ - tell();
  if (avail <= 0)
    return;
  int64_t to_read = std::min(static_cast<int64_t>(READ_BUF_SIZE), avail);
  if (external_file_) {
    ReadExternalDataFile(*external_file_, read_buf_.data(), to_read, external_position_);
    external_position_ += to_read;
    read_buf_end_ = static_cast<size_t>(to_read);
  } else {
    file_stream_.read(reinterpret_cast<char *>(read_buf_.data()), to_read);
    read_buf_end_ = static_cast<size_t>(file_stream_.gcount());
  }
  read_buf_pos_ = 0;
}

void FileStream::_invalidate_read_buffer() {
  if (read_buf_pos_ < read_buf_end_) {
    auto unread = static_cast<std::streamoff>(read_buf_end_ - read_buf_pos_);
    if (external_file_)
      external_position_ -= unread;
    else
      file_stream_.seekg(-unread, std::ios::cur);
    read_buf_pos_ = read_buf_end_ = 0;
  }
}

uint64_t FileStream::next_uint64() {
  // Fast path: single-byte varint already buffered (covers field numbers 1-15).
  if (read_buf_pos_ < read_buf_end_ && (read_buf_[read_buf_pos_] & 0x80) == 0) {
    return static_cast<uint64_t>(read_buf_[read_buf_pos_++]);
  }

  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10; ++i) {
    if (read_buf_pos_ >= read_buf_end_) {
      _fill_read_buffer();
      EXT_ENFORCE(read_buf_pos_ < read_buf_end_,
                  "[FileStream::next_uint64] unable to read an int64 at pos=", tell(),
                  ", size=", size_);
    }
    uint8_t byte = read_buf_[read_buf_pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[FileStream::next_uint64] unable to read an int64 at pos=", tell(), ", size=", size_);
}

const uint8_t *FileStream::read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) {
  EXT_ENFORCE(n_bytes >= 0 && static_cast<int64_t>(tell()) + n_bytes <= size_,
              "[FileStream::read_bytes] out of bounds: tell=", tell(), ", n_bytes=", n_bytes,
              ", size=", size_);
  if (pre_allocated_buffer) {
    // Drain the read-ahead buffer first, then pull any remaining bytes from file.
    size_t buffered = read_buf_end_ - read_buf_pos_;
    size_t from_buf =
        (buffered < static_cast<size_t>(n_bytes)) ? buffered : static_cast<size_t>(n_bytes);
    if (from_buf > 0) {
      memcpy(pre_allocated_buffer, read_buf_.data() + read_buf_pos_, from_buf);
      read_buf_pos_ += from_buf;
    }
    auto remaining = static_cast<offset_t>(n_bytes - from_buf);
    if (remaining > 0) {
      if (external_file_) {
        ReadExternalDataFile(*external_file_, pre_allocated_buffer + from_buf, remaining,
                             external_position_);
        external_position_ += remaining;
        return pre_allocated_buffer;
      }
      file_stream_.read(reinterpret_cast<char *>(pre_allocated_buffer + from_buf), remaining);
      EXT_ENFORCE(file_stream_.gcount() == static_cast<std::streamsize>(remaining),
                  "[FileStream::read_bytes] short read from file: expected=", remaining,
                  ", got=", file_stream_.gcount());
    }
    return pre_allocated_buffer;
  }
  // No pre_allocated_buffer: invalidate the read-ahead buffer so that
  // file_stream_.tellg() equals tell(), then read into the internal buffer_.
  _invalidate_read_buffer();
  if (n_bytes > static_cast<offset_t>(buffer_.size()))
    buffer_.resize(n_bytes);
  if (external_file_) {
    ReadExternalDataFile(*external_file_, buffer_.data(), n_bytes, external_position_);
    external_position_ += n_bytes;
    return buffer_.data();
  }
  file_stream_.read(reinterpret_cast<char *>(buffer_.data()), n_bytes);
  EXT_ENFORCE(file_stream_.gcount() == static_cast<std::streamsize>(n_bytes),
              "[FileStream::read_bytes] short read from file: expected=", n_bytes,
              ", got=", file_stream_.gcount());
  return buffer_.data();
}

void FileStream::skip_bytes(offset_t n_bytes) {
  EXT_ENFORCE(n_bytes >= 0 && static_cast<int64_t>(tell()) + n_bytes <= size_,
              "[FileStream::skip_bytes] out of bounds: tell=", tell(), ", n_bytes=", n_bytes,
              ", size=", size_);
  size_t buffered = read_buf_end_ - read_buf_pos_;
  if (static_cast<size_t>(n_bytes) <= buffered) {
    read_buf_pos_ += static_cast<size_t>(n_bytes);
  } else {
    auto remaining = static_cast<std::streamoff>(n_bytes - buffered);
    read_buf_pos_ = read_buf_end_ = 0;
    if (external_file_)
      external_position_ += remaining;
    else
      file_stream_.seekg(remaining, std::ios::cur);
  }
}

bool FileStream::NotEnd() const { return static_cast<int64_t>(tell()) < size_; }

offset_t FileStream::tell() const {
  if (external_file_)
    return external_position_ - static_cast<offset_t>(read_buf_end_ - read_buf_pos_);
  return static_cast<offset_t>(const_cast<std::ifstream &>(file_stream_).tellg()) -
         static_cast<offset_t>(read_buf_end_ - read_buf_pos_);
}

std::string FileStream::tell_around() const {
  RefString ref(reinterpret_cast<const char *>(buffer_.data()),
                buffer_.size() < 10 ? buffer_.size() : 10);
  return std::string(ref);
}

FileStream::~FileStream() {
#if !defined(_WIN32)
  if (file_descriptor_ >= 0) {
    close(file_descriptor_);
  }
#endif
}

/////////////////
// MmapFileStream
/////////////////

MmapFileStream::MmapFileStream(const std::string &file_path)
    : StringStream(), file_path_(file_path) {
  EXT_ENFORCE(std::filesystem::exists(file_path),
              "MmapFileStream: file does not exist: ", file_path);
  const int64_t fsize = static_cast<int64_t>(std::filesystem::file_size(file_path));
  mmap_ = mmap_file_as_shared_ptr(file_path, fsize);
  // mmap_ is empty when fsize == 0; Setup with a null pointer is fine because
  // size==0 means NotEnd() is immediately false and no read can be issued.
  Setup(mmap_.get(), fsize);
}

void FileStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(block.stream_id == 0,
              "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask([this, block]() {
    if (external_file_) {
      ReadExternalDataFile(*external_file_, block.data, block.size, block.offset);
      return;
    }
#if !defined(_WIN32)
    ReadBlockFromFd(this->file_descriptor_, block, "[FileStream::ReadDelayedBlock]");
#else
    std::ifstream file_stream(this->file_path_, std::ios::binary);
    file_stream.seekg(block.offset);
    file_stream.read(reinterpret_cast<char *>(block.data), block.size);
    EXT_ENFORCE(file_stream.gcount() == static_cast<std::streamsize>(block.size),
                "[FileStream::ReadDelayedBlock] short read: expected=", block.size,
                ", got=", file_stream.gcount());
#endif
  });
  // Advance the stream past the block data, draining the read-ahead buffer first.
  skip_bytes(block.size);
}

void FileStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void FileStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(static_cast<int32_t>(n_threads));
}

int64_t FileStream::TotalDelayedBlockBytes() const {
  int64_t total = 0;
  for (const DelayedBlock &block : blocks_)
    total += static_cast<int64_t>(block.size);
  return total;
}

//////////////////////
// TwoFilesWriteStream
//////////////////////

TwoFilesWriteStream::TwoFilesWriteStream(const std::string &file_path,
                                         const std::string &weights_file)
    : FileWriteStream(file_path),
      weights_stream_(validate_weights_file_is_next_to_model(file_path, weights_file),
                      open_weights_file(file_path, weights_file, true)),
      active_weights_location_(
          std::filesystem::path(weights_stream_.file_path()).filename().string()),
      default_weights_location_(
          std::filesystem::path(weights_stream_.file_path()).filename().string()) {
  std::filesystem::path parent = normalized_model_parent(file_path);
  std::filesystem::path weights = std::filesystem::path(weights_stream_.file_path());
  std::filesystem::path rel =
      std::filesystem::absolute(weights).lexically_normal().lexically_relative(
          std::filesystem::absolute(parent).lexically_normal());
  if (!rel.empty() && rel.string() != ".") {
    default_weights_location_ = rel.string();
  }
  active_weights_location_ = default_weights_location_;
}

void TwoFilesWriteStream::reject_case_aliased_weights_location(const std::string &location) const {
  std::filesystem::path path = std::filesystem::path(location).lexically_normal();
  if (path == std::filesystem::path(weights_stream_.file_path()).lexically_normal()) {
    path = default_weights_location_;
  } else if (path.is_absolute()) {
    path = path.lexically_relative(
        std::filesystem::absolute(normalized_model_parent(file_path_)).lexically_normal());
  }
  const std::string normalized_location = path.string();
  const auto is_case_alias = [&normalized_location](const std::string &existing) {
    const std::string normalized_existing = normalize_external_location(existing);
    return normalized_location != normalized_existing &&
           std::equal(
               normalized_location.begin(), normalized_location.end(), normalized_existing.begin(),
               normalized_existing.end(),
               [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); });
  };
  EXT_ENFORCE(!is_case_alias(default_weights_location_),
              "Case-aliased external weights location: ", location);
  for (const auto &entry : extra_weights_streams_) {
    EXT_ENFORCE(!is_case_alias(entry.first), "Case-aliased external weights location: ", location);
  }
}

void TwoFilesWriteStream::set_active_weights_location(const std::string &location) {
  const std::string normalized_location = normalize_external_location(location);
  if (normalized_location.empty()) {
    active_weights_location_ = weights_stream_.file_path();
    return;
  }
  reject_case_aliased_weights_location(normalized_location);
  if (normalized_location == active_weights_location_) {
    return;
  }
  if (normalized_location == normalize_external_location(weights_stream_.file_path()) ||
      normalized_location == default_weights_location_) {
    active_weights_location_ = normalized_location;
    return;
  }
  auto it = extra_weights_streams_.find(normalized_location);
  if (it == extra_weights_streams_.end()) {
    std::filesystem::path path =
        validate_external_location_is_next_to_model(file_path_, normalized_location);
    auto stream = std::make_unique<FileWriteStream>(
        path.string(),
        OpenExternalDataFile(external_path_to_utf8(normalized_model_parent(file_path_)),
                             normalized_location, true));
    extra_weights_streams_.emplace(normalized_location, std::move(stream));
  }
  active_weights_location_ = normalized_location;
}

void TwoFilesWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  main_buf_.write_raw_bytes(data, n_bytes);
}

int64_t TwoFilesWriteStream::size() const { return main_buf_.size(); }

void TwoFilesWriteStream::FlushMainToFile() {
  int64_t sz = main_buf_.size();
  if (sz > 0) {
    file_stream_.write(reinterpret_cast<const char *>(main_buf_.data()), sz);
    written_bytes_ = static_cast<uint64_t>(sz);
  }
}

void TwoFilesWriteStream::WriteDelayedBlock(DelayedWriteBlock &) {
  EXT_THROW("WriteDelayedBlock is not supported for TwoFilesWriteStream main content; "
            "all large tensor writes are routed to the weights stream.");
}

int64_t TwoFilesWriteStream::weights_size() const {
  if (active_weights_location_ == weights_stream_.file_path() ||
      active_weights_location_ == default_weights_location_) {
    return parallel_write_ ? virtual_write_pos_ : weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  if (parallel_write_) {
    auto pit = extra_virtual_write_pos_.find(active_weights_location_);
    return pit == extra_virtual_write_pos_.end() ? 0 : pit->second;
  }
  return it->second->size();
}

int64_t TwoFilesWriteStream::weights_size(const std::string &location) const {
  const std::string normalized_location = normalize_external_location(location);
  if (normalized_location.empty() ||
      normalized_location == normalize_external_location(weights_stream_.file_path()) ||
      normalized_location == default_weights_location_) {
    return parallel_write_ ? virtual_write_pos_ : weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(normalized_location);
  if (it == extra_weights_streams_.end()) {
    return 0;
  }
  if (parallel_write_) {
    auto pit = extra_virtual_write_pos_.find(normalized_location);
    return pit == extra_virtual_write_pos_.end() ? 0 : pit->second;
  }
  return it->second->size();
}

int64_t TwoFilesWriteStream::weights_size_for_location(const std::string &location) const {
  return weights_size(location);
}

void TwoFilesWriteStream::pre_allocate_weights(int64_t total_bytes) {
  EXT_ENFORCE(total_bytes >= 0, "total_bytes must be non-negative, got ", total_bytes);
  if (total_bytes == 0)
    return;
  weights_stream_.pre_allocate(total_bytes);
}

void TwoFilesWriteStream::pre_allocate_weights(const std::string &location, int64_t total_bytes) {
  EXT_ENFORCE(total_bytes >= 0, "total_bytes must be non-negative, got ", total_bytes);
  if (total_bytes == 0)
    return;
  const std::string normalized_location = normalize_external_location(location);
  if (!normalized_location.empty())
    reject_case_aliased_weights_location(normalized_location);
  if (normalized_location.empty() ||
      normalized_location == normalize_external_location(weights_stream_.file_path()) ||
      normalized_location == default_weights_location_) {
    weights_stream_.pre_allocate(total_bytes);
    return;
  }
  auto it = extra_weights_streams_.find(normalized_location);
  if (it == extra_weights_streams_.end()) {
    std::filesystem::path path =
        validate_external_location_is_next_to_model(file_path_, normalized_location);
    auto stream = std::make_unique<FileWriteStream>(
        path.string(),
        OpenExternalDataFile(external_path_to_utf8(normalized_model_parent(file_path_)),
                             normalized_location, true));
    it = extra_weights_streams_.emplace(normalized_location, std::move(stream)).first;
  }
  it->second->pre_allocate(total_bytes);
}

void TwoFilesWriteStream::StartWriteThreadPool(int32_t n_threads) {
  EXT_ENFORCE(!parallel_write_, "StartWriteThreadPool already called.");
  parallel_write_ = true;
  virtual_write_pos_ = 0;
  extra_virtual_write_pos_.clear();
  write_thread_pool_.Start(n_threads);
}

void TwoFilesWriteStream::WaitForWriteCompletion() {
  if (parallel_write_) {
    write_thread_pool_.Wait();
    parallel_write_ = false;
  }
}

void TwoFilesWriteStream::write_raw_bytes_in_second_stream(const uint8_t *ptr, offset_t n_bytes) {
  if (active_weights_location_ == weights_stream_.file_path() ||
      active_weights_location_ == default_weights_location_) {
    if (parallel_write_) {
      // `virtual_write_pos_` is only ever read and written on the serialization (calling) thread —
      // worker threads capture `offset` by value and never touch `virtual_write_pos_` — so no
      // synchronization is needed here. Every write once StartWriteThreadPool has been called
      // consumes a slice of the virtual offset space, whether it is queued on the thread pool or
      // written inline below the minimum parallel block size, so later writes keep landing at the
      // right offset either way.
      int64_t offset = virtual_write_pos_;
      virtual_write_pos_ += n_bytes;
      // `ptr` points into a TensorProto::raw_data_ vector owned by the ModelProto that was passed
      // to SerializeModelProtoToStream.  WaitForWriteCompletion() is called before that function
      // returns, so the pointed-to memory is guaranteed to outlive every task.
      auto file = weights_stream_.external_file();
      if (n_bytes >= min_parallel_block_size_) {
        write_thread_pool_.SubmitTask(
            [file, ptr, n_bytes, offset]() { WriteExternalDataFile(*file, ptr, n_bytes, offset); });
      } else {
        // Below-threshold writes never run concurrently with another writer of this stream (they
        // execute synchronously on the calling thread), so reuse the already-open weights_stream_
        // file handle instead of opening a brand-new fstream per call.
        weights_stream_.write_raw_bytes_at_offset(ptr, n_bytes, offset);
      }
    } else {
      weights_stream_.write_raw_bytes(ptr, n_bytes);
    }
    return;
  }

  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  if (parallel_write_) {
    // Per-location virtual position; only mutated on the calling thread, same as the
    // default-location path above. Worker threads capture the file and offset by value.
    int64_t &pos = extra_virtual_write_pos_[active_weights_location_];
    int64_t offset = pos;
    pos += n_bytes;
    if (n_bytes >= min_parallel_block_size_) {
      auto file = it->second->external_file();
      write_thread_pool_.SubmitTask(
          [file, ptr, n_bytes, offset]() { WriteExternalDataFile(*file, ptr, n_bytes, offset); });
    } else {
      // Same reasoning as above: below-threshold writes run synchronously on the calling
      // thread, so they can safely reuse the already-open per-location file handle.
      it->second->write_raw_bytes_at_offset(ptr, n_bytes, offset);
    }
  } else {
    it->second->write_raw_bytes(ptr, n_bytes);
  }
}

void TwoFilesWriteStream::write_raw_bytes_in_second_stream(const uint8_t *ptr, offset_t n_bytes,
                                                           const std::string &location) {
  set_active_weights_location(location);
  write_raw_bytes_in_second_stream(ptr, n_bytes);
}

TwoFilesStream::TwoFilesStream(const std::string &file_path, const std::string &weights_file,
                               bool use_mmap_weights)
    : FileStream(file_path),
      weights_stream_(
          std::filesystem::absolute(file_path).lexically_normal() ==
                  std::filesystem::absolute(weights_file).lexically_normal()
              ? FileStream(weights_file)
              : FileStream(weights_file, open_weights_file(file_path, weights_file, false))),
      active_weights_location_(weights_file), default_weights_location_(weights_file),
      use_mmap_weights_(use_mmap_weights) {
  // Automatic discovery uses the model itself as a placeholder, not yet a weights source.
  if (!weights_stream_.external_file()) {
    default_weights_location_ = std::filesystem::path(weights_file).filename().string();
    return;
  }
  default_weights_validated_ = true;
  const auto rel = relative_external_data_path(weights_file, normalized_model_parent(file_path));
  if (!rel.empty() && rel.string() != ".") {
    default_weights_location_ = rel.string();
  }
}

void TwoFilesStream::set_active_weights_location(const std::string &location) {
  const std::string normalized_location = normalize_external_location(location);
  std::filesystem::path model_parent = std::filesystem::path(file_path_).parent_path();
  if (model_parent.empty()) {
    model_parent = std::filesystem::path(".");
  }
  if (normalized_location.empty()) {
    active_weights_location_ = weights_stream_.file_path();
    return;
  }
  if (normalized_location == active_weights_location_) {
    return;
  }
  if (normalized_location == normalize_external_location(weights_stream_.file_path()) ||
      normalized_location == normalize_external_location(default_weights_location_)) {
    active_weights_location_ = normalized_location;
    return;
  }
  auto it = extra_weights_streams_.find(normalized_location);
  if (it == extra_weights_streams_.end()) {
    std::filesystem::path path(normalized_location);
    if (!path.is_absolute()) {
      path = model_parent / path;
    }
    validate_external_weights_read_path(path, model_parent);
    const auto relative = relative_external_data_path(path, model_parent);
    auto stream = std::make_unique<FileStream>(
        path.string(),
        OpenExternalDataFile(external_path_to_utf8(model_parent), external_path_to_utf8(relative)));
    extra_weights_streams_.emplace(normalized_location, std::move(stream));
  }
  active_weights_location_ = normalized_location;
}

bool TwoFilesStream::using_default_weights_location() const {
  const std::string normalized_active_location =
      normalize_external_location(active_weights_location_);
  return normalized_active_location == normalize_external_location(weights_stream_.file_path()) ||
         normalized_active_location == normalize_external_location(default_weights_location_);
}

FileStream &TwoFilesStream::active_weights_stream() {
  if (using_default_weights_location()) {
    if (!default_weights_validated_) {
      auto file = open_weights_file(file_path_, weights_stream_.file_path(), false);
      weights_stream_.file_stream_.close();
#ifndef _WIN32
      ::close(weights_stream_.file_descriptor_);
      weights_stream_.file_descriptor_ = -1;
#endif
      weights_stream_.size_ = ExternalDataFileSize(*file);
      weights_stream_.external_position_ = 0;
      weights_stream_.external_file_ = std::move(file);
      weights_stream_._invalidate_read_buffer();
      default_weights_validated_ = true;
    }
    return weights_stream_;
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  return *it->second;
}

const FileStream &TwoFilesStream::active_weights_stream() const {
  if (using_default_weights_location()) {
    return weights_stream_;
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  return *it->second;
}

int64_t TwoFilesStream::weights_size(const std::string &location) const {
  const std::string normalized_location = normalize_external_location(location);
  if (normalized_location.empty() ||
      normalized_location == normalize_external_location(weights_stream_.file_path()) ||
      normalized_location == normalize_external_location(default_weights_location_)) {
    return weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(normalized_location);
  if (it == extra_weights_streams_.end()) {
    return 0;
  }
  return it->second->size();
}

TwoFilesStream::SharedWeightsBuffer &
TwoFilesStream::ensure_shared_weights_buffer(const std::string &location, size_t alignment) {
  set_active_weights_location(location);
  FileStream &wstream = active_weights_stream();
  const std::string &canonical_path = wstream.file_path();
  auto it = shared_weights_buffers_.find(canonical_path);
  if (it != shared_weights_buffers_.end()) {
    if (alignment > 1) {
      EXT_ENFORCE(it->second.alignment == 0 || it->second.alignment == alignment,
                  "Shared external buffer already loaded with incompatible alignment for file ",
                  canonical_path, ": existing=", it->second.alignment, ", requested=", alignment);
    }
    return it->second;
  }

  SharedWeightsBuffer loaded;
  loaded.size = wstream.size();
  // mmap returns page-aligned memory (typically 4096-byte aligned), which satisfies
  // any tensor alignment up to the page size.  Store alignment=0 to signal that the
  // buffer is compatible with any requested alignment so the compatibility check above
  // always passes on a cache hit.
  loaded.alignment = 0;
  loaded.data = MapExternalDataFile(*wstream.external_file());
  auto inserted = shared_weights_buffers_.emplace(canonical_path, std::move(loaded));
  return inserted.first->second;
}

void TwoFilesStream::read_bytes_from_weights_stream(offset_t n_bytes, uint8_t *pre_allocated_buffer,
                                                    offset_t offset) {
  FileStream &wstream = active_weights_stream();
  if (offset >= 0) {
    // Discard any pre-fetched bytes before seeking so that the read-ahead buffer
    // stays consistent with the file position.
    wstream._invalidate_read_buffer();
    wstream.external_position_ = offset;
  }
  wstream.read_bytes(n_bytes, pre_allocated_buffer);
}

const uint8_t *TwoFilesStream::borrow_weights_bytes(const std::string &location, offset_t offset,
                                                    offset_t n_bytes, size_t alignment,
                                                    std::shared_ptr<void> &owner) {
  EXT_ENFORCE(offset >= 0, "External weights offset must be >= 0, got ", offset);
  EXT_ENFORCE(n_bytes >= 0, "External weights size must be >= 0, got ", n_bytes);
  SharedWeightsBuffer &buffer = ensure_shared_weights_buffer(location, alignment);
  EXT_ENFORCE(offset + n_bytes <= buffer.size, "External weights slice is out of bounds for file ",
              active_weights_stream().file_path(), ": offset=", offset, ", size=", n_bytes,
              ", file_size=", buffer.size);
  owner = std::static_pointer_cast<void>(buffer.data);
  return buffer.data.get() + offset;
}

void TwoFilesStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0 || block.stream_id == 1,
      "Only two streams are allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  if (block.stream_id == 0) {
    thread_pool_.SubmitTask([this, block]() {
      EXT_ENFORCE(block.offset < static_cast<offset_t>(size()),
                  "Offset for weights stream is out of bounds: ", block.offset,
                  " >= ", static_cast<offset_t>(size()));
#if !defined(_WIN32)
      ReadBlockFromFd(this->file_descriptor_, block, "[TwoFilesStream::ReadDelayedBlock#main]");
#else
      std::ifstream file_stream(this->file_path(), std::ios::binary);
      file_stream.seekg(block.offset);
      file_stream.read(reinterpret_cast<char *>(block.data), block.size);
      EXT_ENFORCE(file_stream.gcount() == static_cast<std::streamsize>(block.size),
                  "[TwoFilesStream::ReadDelayedBlock#main] short read: expected=", block.size,
                  ", got=", file_stream.gcount());
#endif
    });
    // Advance past the block, draining the read-ahead buffer first.
    skip_bytes(block.size);
  } else {
    // Snapshot the currently-active weights stream (default or one of the
    // extra weights files) so the worker task always reads from the same
    // file that the caller selected via set_active_weights_location().
    FileStream &target_stream = active_weights_stream();
    const int64_t target_size = target_stream.size();
    auto file = target_stream.external_file();
    thread_pool_.SubmitTask([target_size, file, block]() {
      EXT_ENFORCE(block.offset < static_cast<offset_t>(target_size),
                  "Offset for weights stream is out of bounds: ", block.offset,
                  " >= ", target_size);
      ReadExternalDataFile(*file, block.data, block.size, block.offset);
    });
    // Advance the active weights stream past the block; the weights stream's
    // read buffer is not used for structure parsing, so a direct seek is fine.
    target_stream.external_position_ += static_cast<offset_t>(block.size);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
