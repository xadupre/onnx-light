// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/platform/cpu_descriptor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <filesystem>
#include <set>
#include <sys/auxv.h>
#include <unistd.h>
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define ONNX_LIGHT_CPU_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace ONNX_LIGHT_NAMESPACE::core::platform {
namespace {

constexpr std::array<std::pair<CpuFeature, std::string_view>, 13> kFeatureNames{{
    {CpuFeature::kSse2, "sse2"},
    {CpuFeature::kSse41, "sse4_1"},
    {CpuFeature::kSse42, "sse4_2"},
    {CpuFeature::kAvx, "avx"},
    {CpuFeature::kFma, "fma"},
    {CpuFeature::kAvx2, "avx2"},
    {CpuFeature::kAvx512F, "avx512f"},
    {CpuFeature::kAvx512Bw, "avx512bw"},
    {CpuFeature::kAvx512Vl, "avx512vl"},
    {CpuFeature::kNeon, "neon"},
    {CpuFeature::kDotProduct, "dotprod"},
    {CpuFeature::kSve, "sve"},
    {CpuFeature::kSve2, "sve2"},
}};

std::string CanonicalIdentifier(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  bool previous_separator = false;
  for (unsigned char character : value) {
    if (std::isalnum(character) != 0) {
      result.push_back(static_cast<char>(std::tolower(character)));
      previous_separator = false;
    } else if (!result.empty() && !previous_separator) {
      result.push_back('_');
      previous_separator = true;
    }
  }
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

std::string CanonicalArchitecture(std::string_view value) {
  std::string result = CanonicalIdentifier(value);
  if (result == "amd64" || result == "x64") {
    return "x86_64";
  }
  if (result == "arm64") {
    return "aarch64";
  }
  return result;
}

std::string CanonicalVendor(std::string_view value) {
  std::string result = CanonicalIdentifier(value);
  if (result == "genuineintel") {
    return "intel";
  }
  if (result == "authenticamd") {
    return "amd";
  }
  return result;
}

std::string DetectArchitecture() {
#if defined(_M_X64) || defined(__x86_64__)
  return "x86_64";
#elif defined(_M_IX86) || defined(__i386__)
  return "x86";
#elif defined(_M_ARM64) || defined(__aarch64__)
  return "aarch64";
#elif defined(_M_ARM) || defined(__arm__)
  return "arm";
#elif defined(__riscv) && __riscv_xlen == 64
  return "riscv64";
#elif defined(__powerpc64__) || defined(__ppc64__)
  return "ppc64";
#else
  return "unknown";
#endif
}

std::string DetectMicroarchitecture(std::string_view vendor, std::optional<uint32_t> family,
                                    std::optional<uint32_t> model) {
  if (vendor != "intel" || family != 6 || !model.has_value()) {
    return {};
  }
  switch (*model) {
  case 0x2A:
  case 0x2D:
    return "sandy_bridge";
  case 0x3A:
  case 0x3E:
    return "ivy_bridge";
  case 0x3C:
  case 0x3F:
  case 0x45:
  case 0x46:
    return "haswell";
  case 0x3D:
  case 0x47:
  case 0x4F:
  case 0x56:
    return "broadwell";
  case 0x4E:
  case 0x5E:
    return "skylake";
  case 0x55:
    return "skylake_server";
  case 0x8E:
  case 0x9E:
    return "kaby_lake";
  case 0x66:
    return "cannon_lake";
  case 0x6A:
  case 0x6C:
    return "ice_lake_server";
  case 0x7D:
  case 0x7E:
    return "ice_lake";
  case 0x8C:
  case 0x8D:
    return "tiger_lake";
  case 0x97:
  case 0x9A:
    return "alder_lake";
  case 0xB7:
  case 0xBA:
  case 0xBF:
    return "raptor_lake";
  default:
    return {};
  }
}

#if defined(ONNX_LIGHT_CPU_X86)
struct CpuidRegisters {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
};

bool ReadCpuid(uint32_t leaf, uint32_t subleaf, CpuidRegisters &registers) noexcept {
#if defined(_MSC_VER)
  int values[4]{};
  __cpuidex(values, static_cast<int>(leaf), static_cast<int>(subleaf));
  registers.eax = static_cast<uint32_t>(values[0]);
  registers.ebx = static_cast<uint32_t>(values[1]);
  registers.ecx = static_cast<uint32_t>(values[2]);
  registers.edx = static_cast<uint32_t>(values[3]);
  return true;
#else
  unsigned int maximum_leaf = __get_cpuid_max(leaf & 0x80000000U, nullptr);
  if (maximum_leaf < leaf) {
    return false;
  }
  __cpuid_count(leaf, subleaf, registers.eax, registers.ebx, registers.ecx, registers.edx);
  return true;
#endif
}

uint64_t ReadXcr0() noexcept {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

void DetectX86(CpuDescriptor &descriptor) {
  CpuidRegisters leaf0;
  if (!ReadCpuid(0, 0, leaf0)) {
    return;
  }
  char vendor[13]{};
  std::memcpy(vendor, &leaf0.ebx, sizeof(leaf0.ebx));
  std::memcpy(vendor + 4, &leaf0.edx, sizeof(leaf0.edx));
  std::memcpy(vendor + 8, &leaf0.ecx, sizeof(leaf0.ecx));
  descriptor.vendor = CanonicalVendor(vendor);

  CpuidRegisters leaf1;
  if (!ReadCpuid(1, 0, leaf1)) {
    return;
  }
  uint32_t base_family = (leaf1.eax >> 8) & 0xFU;
  uint32_t base_model = (leaf1.eax >> 4) & 0xFU;
  uint32_t extended_family = (leaf1.eax >> 20) & 0xFFU;
  uint32_t extended_model = (leaf1.eax >> 16) & 0xFU;
  descriptor.family = base_family == 0xFU ? base_family + extended_family : base_family;
  descriptor.model = (base_family == 0x6U || base_family == 0xFU)
                         ? base_model + (extended_model << 4)
                         : base_model;
  descriptor.stepping = leaf1.eax & 0xFU;

  if ((leaf1.edx & (1U << 26)) != 0) {
    descriptor.features.Add(CpuFeature::kSse2);
  }
  if ((leaf1.ecx & (1U << 19)) != 0) {
    descriptor.features.Add(CpuFeature::kSse41);
  }
  if ((leaf1.ecx & (1U << 20)) != 0) {
    descriptor.features.Add(CpuFeature::kSse42);
  }

  bool os_xsave = (leaf1.ecx & (1U << 27)) != 0;
  bool hardware_avx = (leaf1.ecx & (1U << 28)) != 0;
  uint64_t xcr0 = os_xsave ? ReadXcr0() : 0;
  bool avx_usable = hardware_avx && (xcr0 & 0x6U) == 0x6U;
  if (avx_usable) {
    descriptor.features.Add(CpuFeature::kAvx);
    if ((leaf1.ecx & (1U << 12)) != 0) {
      descriptor.features.Add(CpuFeature::kFma);
    }
  }

  CpuidRegisters leaf7;
  if (!ReadCpuid(7, 0, leaf7)) {
    return;
  }
  if (avx_usable && (leaf7.ebx & (1U << 5)) != 0) {
    descriptor.features.Add(CpuFeature::kAvx2);
  }
  bool avx512_usable = avx_usable && (xcr0 & 0xE0U) == 0xE0U;
  if (avx512_usable && (leaf7.ebx & (1U << 16)) != 0) {
    descriptor.features.Add(CpuFeature::kAvx512F);
  }
  if (avx512_usable && (leaf7.ebx & (1U << 30)) != 0) {
    descriptor.features.Add(CpuFeature::kAvx512Bw);
  }
  if (avx512_usable && (leaf7.ebx & (1U << 31)) != 0) {
    descriptor.features.Add(CpuFeature::kAvx512Vl);
  }
}
#endif

#if defined(__linux__)
void SetPositive(std::optional<size_t> &destination, long value) {
  if (value > 0) {
    destination = static_cast<size_t>(value);
  }
}

std::optional<uint32_t> ReadUnsignedFile(const std::filesystem::path &path) {
  std::ifstream stream(path);
  std::string value;
  if (!stream || !std::getline(stream, value)) {
    return std::nullopt;
  }
  uint32_t parsed = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return parsed;
}

void DetectLinuxTopology(CpuDescriptor &descriptor) {
  std::error_code error;
  std::filesystem::directory_iterator iterator("/sys/devices/system/cpu", error);
  std::filesystem::directory_iterator end;
  std::set<std::pair<uint32_t, uint32_t>> physical_cores;
  uint32_t logical_cores = 0;
  while (!error && iterator != end) {
    const std::filesystem::path cpu_path = iterator->path();
    std::string name = cpu_path.filename().string();
    bool cpu_directory = name.size() > 3 && name.starts_with("cpu") &&
                         std::all_of(name.begin() + 3, name.end(), [](unsigned char character) {
                           return std::isdigit(character) != 0;
                         });
    iterator.increment(error);
    if (!cpu_directory) {
      continue;
    }
    std::optional<uint32_t> online = ReadUnsignedFile(cpu_path / "online");
    if (online.has_value() && *online == 0) {
      continue;
    }
    ++logical_cores;
    std::optional<uint32_t> core_id = ReadUnsignedFile(cpu_path / "topology/core_id");
    std::optional<uint32_t> package_id =
        ReadUnsignedFile(cpu_path / "topology/physical_package_id");
    if (core_id.has_value() && package_id.has_value()) {
      physical_cores.emplace(*package_id, *core_id);
    }
  }
  if (logical_cores != 0) {
    descriptor.logical_cores = logical_cores;
  }
  if (!physical_cores.empty()) {
    descriptor.physical_cores = static_cast<uint32_t>(physical_cores.size());
  }
}

#if defined(__aarch64__) || defined(__arm__)
std::optional<uint32_t> ParseUnsignedValue(std::string_view text, int base) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
  }
  uint32_t value = 0;
  const char *begin = text.data();
  const char *end = begin + text.size();
  auto result = std::from_chars(begin, end, value, base);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return value;
}

std::string ArmVendor(uint32_t implementer) {
  switch (implementer) {
  case 0x41:
    return "arm";
  case 0x42:
    return "broadcom";
  case 0x43:
    return "cavium";
  case 0x50:
    return "ampere";
  case 0x51:
    return "qualcomm";
  case 0x61:
    return "apple";
  default:
    return {};
  }
}

void DetectLinuxArmIdentity(CpuDescriptor &descriptor) {
  std::ifstream stream("/proc/cpuinfo");
  std::string line;
  while (std::getline(stream, line)) {
    size_t separator = line.find(':');
    if (separator == std::string::npos) {
      continue;
    }
    std::string key = CanonicalIdentifier(std::string_view(line).substr(0, separator));
    std::string_view value(line.data() + separator + 1, line.size() - separator - 1);
    int base = key == "cpu_implementer" || key == "cpu_part" ? 16 : 10;
    std::optional<uint32_t> parsed = ParseUnsignedValue(value, base);
    if (!parsed.has_value()) {
      continue;
    }
    if (key == "cpu_implementer") {
      descriptor.vendor = ArmVendor(*parsed);
    } else if (key == "cpu_architecture") {
      descriptor.family = *parsed;
    } else if (key == "cpu_part") {
      descriptor.model = *parsed;
    } else if (key == "cpu_revision") {
      descriptor.stepping = *parsed;
    }
    if (!descriptor.vendor.empty() && descriptor.family.has_value() &&
        descriptor.model.has_value() && descriptor.stepping.has_value()) {
      break;
    }
  }
}

void DetectLinuxArmFeatures(CpuDescriptor &descriptor) {
  unsigned long capabilities = getauxval(AT_HWCAP);
  unsigned long capabilities2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_ASIMD
  if ((capabilities & HWCAP_ASIMD) != 0) {
    descriptor.features.Add(CpuFeature::kNeon);
  }
#elif defined(HWCAP_NEON)
  if ((capabilities & HWCAP_NEON) != 0) {
    descriptor.features.Add(CpuFeature::kNeon);
  }
#endif
#ifdef HWCAP_ASIMDDP
  if ((capabilities & HWCAP_ASIMDDP) != 0) {
    descriptor.features.Add(CpuFeature::kDotProduct);
  }
#endif
#ifdef HWCAP_SVE
  if ((capabilities & HWCAP_SVE) != 0) {
    descriptor.features.Add(CpuFeature::kSve);
  }
#endif
#ifdef HWCAP2_SVE2
  if ((capabilities2 & HWCAP2_SVE2) != 0) {
    descriptor.features.Add(CpuFeature::kSve2);
  }
#else
  (void)capabilities2;
#endif
}
#endif

void DetectLinux(CpuDescriptor &descriptor) {
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
  SetPositive(descriptor.cache_line_bytes, sysconf(_SC_LEVEL1_DCACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL1_DCACHE_SIZE
  SetPositive(descriptor.l1_data_bytes, sysconf(_SC_LEVEL1_DCACHE_SIZE));
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
  SetPositive(descriptor.l2_bytes, sysconf(_SC_LEVEL2_CACHE_SIZE));
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
  SetPositive(descriptor.l3_bytes, sysconf(_SC_LEVEL3_CACHE_SIZE));
#endif
  DetectLinuxTopology(descriptor);
#if defined(__aarch64__) || defined(__arm__)
  DetectLinuxArmIdentity(descriptor);
  DetectLinuxArmFeatures(descriptor);
#endif
}
#endif

#if defined(__APPLE__)
template <typename T> std::optional<T> ReadSysctlValue(const char *name) {
  T value{};
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || size != sizeof(value)) {
    return std::nullopt;
  }
  return value;
}

void DetectApple(CpuDescriptor &descriptor) {
  descriptor.cache_line_bytes = ReadSysctlValue<size_t>("hw.cachelinesize");
  descriptor.l1_data_bytes = ReadSysctlValue<size_t>("hw.l1dcachesize");
  descriptor.l2_bytes = ReadSysctlValue<size_t>("hw.l2cachesize");
  descriptor.l3_bytes = ReadSysctlValue<size_t>("hw.l3cachesize");
  descriptor.physical_cores = ReadSysctlValue<uint32_t>("hw.physicalcpu");
  descriptor.logical_cores = ReadSysctlValue<uint32_t>("hw.logicalcpu");
#if defined(__aarch64__) || defined(_M_ARM64)
  descriptor.vendor = "apple";
  int neon = ReadSysctlValue<int>("hw.optional.neon")
                 .value_or(ReadSysctlValue<int>("hw.optional.AdvSIMD").value_or(0));
  if (neon != 0) {
    descriptor.features.Add(CpuFeature::kNeon);
  }
  if (ReadSysctlValue<int>("hw.optional.arm.FEAT_DotProd").value_or(0) != 0) {
    descriptor.features.Add(CpuFeature::kDotProduct);
  }
#endif
}
#endif

#if defined(_WIN32)
void DetectWindows(CpuDescriptor &descriptor) {
  DWORD byte_count = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &byte_count);
  if (byte_count == 0) {
    return;
  }
  std::vector<std::byte> buffer(byte_count);
  auto *information = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationAll, information, &byte_count)) {
    return;
  }

  uint32_t physical_cores = 0;
  uint32_t logical_cores = 0;
  size_t offset = 0;
  while (offset < byte_count) {
    auto *entry =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
    if (entry->Relationship == RelationProcessorCore) {
      ++physical_cores;
      for (WORD group = 0; group < entry->Processor.GroupCount; ++group) {
        logical_cores += static_cast<uint32_t>(
            std::popcount(static_cast<uint64_t>(entry->Processor.GroupMask[group].Mask)));
      }
    } else if (entry->Relationship == RelationCache) {
      const CACHE_RELATIONSHIP &cache = entry->Cache;
      if (cache.LineSize != 0 && (!descriptor.cache_line_bytes.has_value() ||
                                  cache.LineSize > *descriptor.cache_line_bytes)) {
        descriptor.cache_line_bytes = cache.LineSize;
      }
      if (cache.Type == CacheData || cache.Type == CacheUnified) {
        std::optional<size_t> *destination = nullptr;
        if (cache.Level == 1) {
          destination = &descriptor.l1_data_bytes;
        } else if (cache.Level == 2) {
          destination = &descriptor.l2_bytes;
        } else if (cache.Level == 3) {
          destination = &descriptor.l3_bytes;
        }
        if (destination != nullptr &&
            (!destination->has_value() || cache.CacheSize > **destination)) {
          *destination = cache.CacheSize;
        }
      }
    }
    if (entry->Size == 0) {
      break;
    }
    offset += entry->Size;
  }
  if (physical_cores != 0) {
    descriptor.physical_cores = physical_cores;
  }
  if (logical_cores != 0) {
    descriptor.logical_cores = logical_cores;
  }
#if defined(_M_ARM64) || defined(_M_ARM)
  if (IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE)) {
    descriptor.features.Add(CpuFeature::kNeon);
  }
#endif
}
#endif

} // namespace

std::string_view CpuFeatureName(CpuFeature feature) noexcept {
  auto found = std::find_if(kFeatureNames.begin(), kFeatureNames.end(),
                            [feature](const auto &entry) { return entry.first == feature; });
  return found == kFeatureNames.end() ? std::string_view{} : found->second;
}

std::optional<CpuFeature> CpuFeatureFromName(std::string_view name) noexcept {
  auto found = std::find_if(kFeatureNames.begin(), kFeatureNames.end(),
                            [name](const auto &entry) { return entry.second == name; });
  return found == kFeatureNames.end() ? std::nullopt : std::optional<CpuFeature>(found->first);
}

CpuDescriptor DetectCpuDescriptor() {
  CpuDescriptor descriptor;
  descriptor.architecture = DetectArchitecture();

#if defined(ONNX_LIGHT_CPU_X86)
  DetectX86(descriptor);
#endif
#if defined(__linux__)
  DetectLinux(descriptor);
#elif defined(__APPLE__)
  DetectApple(descriptor);
#elif defined(_WIN32)
  DetectWindows(descriptor);
#endif

  if (!descriptor.logical_cores.has_value()) {
    unsigned int logical_cores = std::thread::hardware_concurrency();
    if (logical_cores != 0) {
      descriptor.logical_cores = logical_cores;
    }
  }
  descriptor.vendor = CanonicalVendor(descriptor.vendor);
  if (descriptor.microarchitecture.empty()) {
    descriptor.microarchitecture =
        DetectMicroarchitecture(descriptor.vendor, descriptor.family, descriptor.model);
  }
  return descriptor;
}

const CpuDescriptor &GetCpuDescriptor() {
  static const CpuDescriptor descriptor = DetectCpuDescriptor();
  return descriptor;
}

bool CpuSelector::Matches(const CpuDescriptor &processor,
                          std::optional<uint32_t> effective_threads) const {
  if (architecture.has_value() &&
      CanonicalArchitecture(*architecture) != CanonicalArchitecture(processor.architecture)) {
    return false;
  }
  if (vendor.has_value() && CanonicalVendor(*vendor) != CanonicalVendor(processor.vendor)) {
    return false;
  }
  if (family.has_value() && processor.family != family) {
    return false;
  }
  if (!models.empty() &&
      (!processor.model.has_value() ||
       std::find(models.begin(), models.end(), *processor.model) == models.end())) {
    return false;
  }
  if (microarchitecture.has_value() &&
      CanonicalIdentifier(*microarchitecture) != CanonicalIdentifier(processor.microarchitecture)) {
    return false;
  }
  if (!processor.features.ContainsAll(required_features) ||
      processor.features.Intersects(excluded_features)) {
    return false;
  }
  if ((minimum_threads.has_value() || maximum_threads.has_value()) &&
      !effective_threads.has_value()) {
    return false;
  }
  if (minimum_threads.has_value() && *effective_threads < *minimum_threads) {
    return false;
  }
  if (maximum_threads.has_value() && *effective_threads > *maximum_threads) {
    return false;
  }
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::platform
