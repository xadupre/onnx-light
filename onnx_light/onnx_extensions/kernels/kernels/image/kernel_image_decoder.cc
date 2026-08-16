// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

using core::runtime::RawBuffer;
using core::runtime::RawByteBuffer;

void CheckImageDecoderInput(const Tensor &encoded_stream) {
  EXT_ENFORCE_INVALID(encoded_stream.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::ImageDecoder only supports UINT8 input tensors.");
  EXT_ENFORCE_INVALID(encoded_stream.shape.size() == 1u,
                      "kernel::ImageDecoder input ``encoded_stream`` must be a 1-D tensor "
                      "carrying the encoded bytestream.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(encoded_stream.size_bytes()) ==
                          encoded_stream.element_count(),
                      "kernel::ImageDecoder input ``encoded_stream`` data size does not "
                      "match its shape.");
}

// ---------------------------------------------------------------------------
// BMP decoder
//
// Decodes an uncompressed 24-bit BMP bytestream (BI_RGB, BITMAPINFOHEADER)
// without any external library dependency. Returns true on success and fills
// ``pixels`` with the channel-last RGB (or BGR / Grayscale) pixel data; on
// failure returns false so the caller can fall back to the empty-matrix path.
// ---------------------------------------------------------------------------

inline uint16_t ReadU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t ReadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline int32_t ReadI32LE(const uint8_t *p) { return static_cast<int32_t>(ReadU32LE(p)); }

bool TryDecodeBmp(const uint8_t *data, size_t size, const std::string &pixel_format,
                  int64_t &out_height, int64_t &out_width, RawByteBuffer &out_pixels) {
  // Minimum BMP file: 14-byte file header + 40-byte BITMAPINFOHEADER = 54 bytes.
  if (size < 54) {
    return false;
  }
  // Check "BM" magic.
  if (data[0] != 0x42 || data[1] != 0x4D) {
    return false;
  }

  const uint32_t pixel_offset = ReadU32LE(data + 10);
  const uint32_t dib_header_size = ReadU32LE(data + 14);

  // Only BITMAPINFOHEADER (size==40) is supported; BITMAPCOREHEADER (size==12)
  // and extended variants are not encountered in the backend test data.
  if (dib_header_size < 40) {
    return false;
  }

  const int32_t width_signed = ReadI32LE(data + 18);
  const int32_t height_signed = ReadI32LE(data + 22);
  const uint16_t bpp = ReadU16LE(data + 28);
  const uint32_t compression = ReadU32LE(data + 30);

  // Only uncompressed 24-bit BGR is supported.
  if (bpp != 24 || compression != 0) {
    return false;
  }
  if (width_signed <= 0) {
    return false;
  }

  const int32_t width = width_signed;
  // Negative height means top-down (rows are already in display order).
  const bool bottom_up = height_signed > 0;
  const int32_t height = bottom_up ? height_signed : -height_signed;

  if (height <= 0) {
    return false;
  }

  // Row stride is padded to a 4-byte boundary.
  const uint32_t row_stride = (static_cast<uint32_t>(width) * 3u + 3u) & ~3u;
  const uint64_t required_bytes =
      static_cast<uint64_t>(pixel_offset) + static_cast<uint64_t>(row_stride) * height;
  if (required_bytes > size) {
    return false;
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  const size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
  out_pixels.resize(total_pixels * static_cast<size_t>(channels));

  const uint8_t *pixel_data = data + pixel_offset;
  for (int32_t row = 0; row < height; ++row) {
    // BMP rows are stored bottom-to-top when height is positive.
    const int32_t src_row = bottom_up ? (height - 1 - row) : row;
    const uint8_t *src_row_ptr = pixel_data + static_cast<size_t>(src_row) * row_stride;
    uint8_t *dst_row_ptr =
        out_pixels.data() + static_cast<size_t>(row) * static_cast<size_t>(width) * channels;

    for (int32_t col = 0; col < width; ++col) {
      const uint8_t b = src_row_ptr[col * 3 + 0];
      const uint8_t g = src_row_ptr[col * 3 + 1];
      const uint8_t r = src_row_ptr[col * 3 + 2];
      uint8_t *dst = dst_row_ptr + col * channels;
      if (pixel_format == "RGB") {
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
      } else if (pixel_format == "BGR") {
        dst[0] = b;
        dst[1] = g;
        dst[2] = r;
      } else {
        // Grayscale: standard luminance formula (ITU-R BT.601).
        dst[0] = static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000);
      }
    }
  }

  out_height = static_cast<int64_t>(height);
  out_width = static_cast<int64_t>(width);
  return true;
}

// ---------------------------------------------------------------------------
// Big-endian byte readers (used by the PNG decoder below).
// ---------------------------------------------------------------------------
inline uint16_t ReadU16BE(const uint8_t *p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

inline uint32_t ReadU32BE(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

// ---------------------------------------------------------------------------
// Baseline JPEG (JFIF, SOF0) decoder
//
// Implements the subset of the JPEG standard needed to decode the small
// baseline JFIF bytestreams emitted by Pillow / OpenCV for the backend
// test cases (8-bit precision, sequential baseline, 1 or 3 components,
// horizontal/vertical sampling factors in {1, 2}, optional restart
// intervals). Progressive, arithmetic-coded and 12-bit JPEGs are out of
// scope and fall through to the empty-matrix path.
// ---------------------------------------------------------------------------

// Inverse zig-zag: ``kZigZagInverse[k]`` is the natural-order row-major
// 8x8 position of the k-th coefficient in zig-zag scan order.
constexpr std::array<uint8_t, 64> kZigZagInverse = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

struct HuffmanTable {
  std::array<int32_t, 16> max_code{};
  std::array<int32_t, 16> min_code{};
  std::array<int32_t, 16> value_offset{};
  std::vector<uint8_t> values;
  bool built = false;
};

struct ComponentInfo {
  int id = 0;
  int h_sampling = 1;
  int v_sampling = 1;
  int quant_table_id = 0;
  int dc_table_id = 0;
  int ac_table_id = 0;
  int32_t prev_dc = 0;
  std::unique_ptr<detail::TemporaryTypedBuffer<uint8_t>> samples;
  int sample_width = 0;
  int sample_height = 0;
};

class JpegBitReader {
public:
  JpegBitReader(const uint8_t *data, size_t size) : data_(data), size_(size) {}

  int32_t ReadBits(int n) {
    while (bit_count_ < n) {
      if (eof_) {
        return -1;
      }
      uint8_t byte;
      if (!NextByte(byte)) {
        return -1;
      }
      bit_buffer_ = (bit_buffer_ << 8) | byte;
      bit_count_ += 8;
    }
    const int32_t result =
        static_cast<int32_t>((bit_buffer_ >> (bit_count_ - n)) & ((1u << n) - 1u));
    bit_count_ -= n;
    return result;
  }

  // Discards any partial byte and resets the saw-restart flag so the
  // caller can resume reading after a restart-marker resync.
  void ResetForRestart() {
    bit_buffer_ = 0;
    bit_count_ = 0;
    saw_restart_ = false;
    eof_ = false;
  }

  bool saw_restart() const { return saw_restart_; }
  bool eof() const { return eof_; }
  size_t position() const { return pos_; }

private:
  bool NextByte(uint8_t &out) {
    if (pos_ >= size_) {
      eof_ = true;
      return false;
    }
    uint8_t b = data_[pos_++];
    if (b != 0xFF) {
      out = b;
      return true;
    }
    // 0xFF byte: skip 0xFF padding then inspect the next byte.
    while (pos_ < size_ && data_[pos_] == 0xFF) {
      ++pos_;
    }
    if (pos_ >= size_) {
      eof_ = true;
      return false;
    }
    uint8_t next = data_[pos_++];
    if (next == 0x00) {
      out = 0xFF;
      return true;
    }
    // Embedded marker (restart or EOI): stop and report to caller.
    saw_restart_ = true;
    eof_ = true;
    return false;
  }

  const uint8_t *data_;
  size_t size_;
  size_t pos_ = 0;
  uint32_t bit_buffer_ = 0;
  int bit_count_ = 0;
  bool eof_ = false;
  bool saw_restart_ = false;
};

int32_t DecodeHuffman(JpegBitReader &reader, const HuffmanTable &table) {
  if (!table.built) {
    return -1;
  }
  int32_t code = 0;
  for (int len = 0; len < 16; ++len) {
    const int32_t bit = reader.ReadBits(1);
    if (bit < 0) {
      return -1;
    }
    code = (code << 1) | bit;
    if (code <= table.max_code[len]) {
      const int32_t idx = table.value_offset[len] + code;
      if (idx < 0 || idx >= static_cast<int32_t>(table.values.size())) {
        return -1;
      }
      return table.values[idx];
    }
  }
  return -1;
}

bool BuildHuffmanTable(const uint8_t *bits, const uint8_t *values, size_t value_count,
                       HuffmanTable &table) {
  table.values.assign(values, values + value_count);
  int32_t code = 0;
  int32_t offset = 0;
  for (int len = 0; len < 16; ++len) {
    const int n = bits[len];
    table.value_offset[len] = offset - code;
    table.min_code[len] = code;
    code += n;
    table.max_code[len] = n ? code - 1 : -1;
    code <<= 1;
    offset += n;
  }
  if (offset != static_cast<int32_t>(value_count)) {
    return false;
  }
  table.built = true;
  return true;
}

// Sign-extends an ``n``-bit magnitude category code per CCITT T.81 Fig F.12.
inline int32_t JpegExtend(int32_t value, int n) {
  if (n == 0) {
    return 0;
  }
  const int32_t vt = 1 << (n - 1);
  return value < vt ? value + static_cast<int32_t>(static_cast<uint32_t>(-1) << n) + 1 : value;
}

// Reference floating-point 8x8 IDCT with level shift and clamping.
// Operates on dequantized coefficients in natural row-major order.
void IdctBlock(const std::array<double, 64> &in, std::array<uint8_t, 64> &out) {
  static const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  constexpr double kPi = 3.14159265358979323846;
  std::array<double, 64> tmp{};
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      double sum = 0.0;
      for (int u = 0; u < 8; ++u) {
        const double cu = u == 0 ? inv_sqrt2 : 1.0;
        sum += cu * in[u * 8 + x] * std::cos((2 * y + 1) * u * kPi / 16.0);
      }
      tmp[y * 8 + x] = sum;
    }
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      double sum = 0.0;
      for (int v = 0; v < 8; ++v) {
        const double cv = v == 0 ? inv_sqrt2 : 1.0;
        sum += cv * tmp[y * 8 + v] * std::cos((2 * x + 1) * v * kPi / 16.0);
      }
      const double value = sum * 0.25 + 128.0;
      const int32_t r = static_cast<int32_t>(std::lround(value));
      out[y * 8 + x] = static_cast<uint8_t>(r < 0 ? 0 : r > 255 ? 255 : r);
    }
  }
}

// Decodes a single 8x8 block: DPCM-decode the DC coefficient and the
// run-length / category encoded AC coefficients, dequantize using
// ``qtable`` and IDCT into ``block_samples``. Returns false on stream
// errors so the caller can abort cleanly.
bool DecodeBlock(JpegBitReader &reader, const HuffmanTable &dc_table, const HuffmanTable &ac_table,
                 const std::array<uint16_t, 64> &qtable, int32_t &prev_dc,
                 std::array<uint8_t, 64> &block_samples) {
  std::array<double, 64> coeffs{};

  const int32_t dc_cat = DecodeHuffman(reader, dc_table);
  if (dc_cat < 0 || dc_cat > 15) {
    return false;
  }
  int32_t diff = 0;
  if (dc_cat > 0) {
    const int32_t raw = reader.ReadBits(dc_cat);
    if (raw < 0) {
      return false;
    }
    diff = JpegExtend(raw, dc_cat);
  }
  prev_dc += diff;
  coeffs[0] = static_cast<double>(prev_dc) * static_cast<double>(qtable[0]);

  int k = 1;
  while (k < 64) {
    const int32_t rs = DecodeHuffman(reader, ac_table);
    if (rs < 0) {
      return false;
    }
    if (rs == 0x00) {
      break; // End-Of-Block: remaining coefficients are zero.
    }
    if (rs == 0xF0) {
      k += 16; // ZRL: 16 zeros in a row.
      continue;
    }
    const int run = rs >> 4;
    const int size = rs & 0x0F;
    if (size == 0) {
      return false;
    }
    k += run;
    if (k >= 64) {
      return false;
    }
    const int32_t raw = reader.ReadBits(size);
    if (raw < 0) {
      return false;
    }
    const int32_t value = JpegExtend(raw, size);
    const int natural = kZigZagInverse[k];
    coeffs[natural] = static_cast<double>(value) * static_cast<double>(qtable[k]);
    ++k;
  }
  IdctBlock(coeffs, block_samples);
  return true;
}

bool TryDecodeJpeg(const uint8_t *data, size_t size, const std::string &pixel_format,
                   int64_t &out_height, int64_t &out_width, RawByteBuffer &out_pixels,
                   RawBufferAllocator *allocator) {
  // SOI marker.
  if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
    return false;
  }

  size_t pos = 2;
  // Per-table state.
  std::array<std::array<uint16_t, 64>, 4> qtables{};
  std::array<bool, 4> qtable_present{};
  std::array<HuffmanTable, 4> dc_tables;
  std::array<HuffmanTable, 4> ac_tables;
  int frame_width = 0;
  int frame_height = 0;
  int num_components = 0;
  std::array<ComponentInfo, 4> components{};
  int restart_interval = 0;
  bool sof_seen = false;
  size_t scan_start = 0;
  size_t scan_end = 0;
  std::array<int, 4> scan_components{};
  int scan_count = 0;

  while (pos + 2 <= size) {
    if (data[pos] != 0xFF) {
      return false;
    }
    // Skip filler 0xFF bytes.
    while (pos < size && data[pos] == 0xFF) {
      ++pos;
    }
    if (pos >= size) {
      return false;
    }
    const uint8_t marker = data[pos++];
    if (marker == 0xD9) { // EOI
      break;
    }
    // Standalone markers without a length field: RST0..RST7 (handled
    // inside the scan), TEM. We ignore TEM here.
    if (marker == 0x01) {
      continue;
    }
    if (pos + 2 > size) {
      return false;
    }
    const size_t seg_len =
        (static_cast<size_t>(data[pos]) << 8) | static_cast<size_t>(data[pos + 1]);
    if (seg_len < 2 || pos + seg_len > size) {
      return false;
    }
    const uint8_t *seg = data + pos + 2;
    const size_t seg_size = seg_len - 2;
    pos += seg_len;

    if (marker == 0xC0) { // SOF0 baseline
      if (seg_size < 6) {
        return false;
      }
      if (seg[0] != 8) {
        // 12-bit precision not supported.
        return false;
      }
      frame_height = (static_cast<int>(seg[1]) << 8) | seg[2];
      frame_width = (static_cast<int>(seg[3]) << 8) | seg[4];
      num_components = seg[5];
      if (num_components != 1 && num_components != 3) {
        return false;
      }
      if (seg_size < 6 + static_cast<size_t>(num_components) * 3u) {
        return false;
      }
      for (int i = 0; i < num_components; ++i) {
        const uint8_t *cp = seg + 6 + i * 3;
        components[i].id = cp[0];
        components[i].h_sampling = (cp[1] >> 4) & 0x0F;
        components[i].v_sampling = cp[1] & 0x0F;
        components[i].quant_table_id = cp[2];
        if (components[i].h_sampling < 1 || components[i].h_sampling > 2 ||
            components[i].v_sampling < 1 || components[i].v_sampling > 2 ||
            components[i].quant_table_id > 3) {
          return false;
        }
      }
      sof_seen = true;
    } else if (marker == 0xDB) { // DQT
      size_t p = 0;
      while (p < seg_size) {
        const uint8_t tq = seg[p++];
        const int precision = (tq >> 4) & 0x0F;
        const int id = tq & 0x0F;
        if (id > 3) {
          return false;
        }
        const size_t bytes = (precision == 0) ? 64u : 128u;
        if (precision > 1 || p + bytes > seg_size) {
          return false;
        }
        for (int k = 0; k < 64; ++k) {
          uint16_t value;
          if (precision == 0) {
            value = seg[p++];
          } else {
            value = static_cast<uint16_t>((static_cast<uint16_t>(seg[p]) << 8) | seg[p + 1]);
            p += 2;
          }
          qtables[id][k] = value;
        }
        qtable_present[id] = true;
      }
    } else if (marker == 0xC4) { // DHT
      size_t p = 0;
      while (p < seg_size) {
        if (p + 17 > seg_size) {
          return false;
        }
        const uint8_t tc = seg[p++];
        const int klass = (tc >> 4) & 0x0F;
        const int id = tc & 0x0F;
        if (klass > 1 || id > 3) {
          return false;
        }
        const uint8_t *bits = seg + p;
        p += 16;
        size_t total = 0;
        for (int i = 0; i < 16; ++i) {
          total += bits[i];
        }
        if (p + total > seg_size || total > 256) {
          return false;
        }
        HuffmanTable &target = (klass == 0) ? dc_tables[id] : ac_tables[id];
        if (!BuildHuffmanTable(bits, seg + p, total, target)) {
          return false;
        }
        p += total;
      }
    } else if (marker == 0xDD) { // DRI
      if (seg_size != 2) {
        return false;
      }
      restart_interval = (static_cast<int>(seg[0]) << 8) | seg[1];
    } else if (marker == 0xDA) { // SOS
      if (seg_size < 1) {
        return false;
      }
      scan_count = seg[0];
      if (scan_count < 1 || scan_count > 4 ||
          seg_size < 1 + static_cast<size_t>(scan_count) * 2u + 3u) {
        return false;
      }
      for (int i = 0; i < scan_count; ++i) {
        const uint8_t comp_id = seg[1 + i * 2];
        const uint8_t tdta = seg[1 + i * 2 + 1];
        // Locate the matching component declared in SOF0.
        int comp_idx = -1;
        for (int c = 0; c < num_components; ++c) {
          if (components[c].id == comp_id) {
            comp_idx = c;
            break;
          }
        }
        if (comp_idx < 0) {
          return false;
        }
        components[comp_idx].dc_table_id = (tdta >> 4) & 0x0F;
        components[comp_idx].ac_table_id = tdta & 0x0F;
        scan_components[i] = comp_idx;
      }
      // Baseline sequential: Ss=0, Se=63, Ah=Al=0. We do not validate
      // those bytes strictly since the only baseline encoders we
      // target always emit them with these canonical values.
      scan_start = pos;
      scan_end = size;
      sof_seen = sof_seen; // silence unused warning when no other branches
      break;
    } else if (marker >= 0xE0 && marker <= 0xEF) {
      // APPn segments (JFIF, Exif, etc.): ignored.
    } else if (marker == 0xFE) {
      // Comment: ignored.
    } else if (marker >= 0xC1 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8) {
      // Non-baseline SOF (progressive, lossless, etc.): out of scope.
      return false;
    } else {
      // Other markers (e.g. DAC, DNL, JPG): skip silently.
    }
  }

  if (!sof_seen || scan_count == 0 || frame_width <= 0 || frame_height <= 0) {
    return false;
  }
  for (int i = 0; i < scan_count; ++i) {
    const ComponentInfo &c = components[scan_components[i]];
    if (!qtable_present[c.quant_table_id] || !dc_tables[c.dc_table_id].built ||
        !ac_tables[c.ac_table_id].built) {
      return false;
    }
  }

  // Compute maximum sampling factors and MCU layout.
  int max_h = 1, max_v = 1;
  for (int i = 0; i < num_components; ++i) {
    if (components[i].h_sampling > max_h) {
      max_h = components[i].h_sampling;
    }
    if (components[i].v_sampling > max_v) {
      max_v = components[i].v_sampling;
    }
  }
  const int mcu_w = max_h * 8;
  const int mcu_h = max_v * 8;
  const int mcu_cols = (frame_width + mcu_w - 1) / mcu_w;
  const int mcu_rows = (frame_height + mcu_h - 1) / mcu_h;

  for (int i = 0; i < num_components; ++i) {
    ComponentInfo &c = components[i];
    c.sample_width = mcu_cols * c.h_sampling * 8;
    c.sample_height = mcu_rows * c.v_sampling * 8;
    c.samples = std::make_unique<detail::TemporaryTypedBuffer<uint8_t>>(
        static_cast<size_t>(c.sample_width) * c.sample_height, allocator, "JPEG samples");
    c.prev_dc = 0;
  }

  JpegBitReader reader(data + scan_start, scan_end - scan_start);
  int mcu_index = 0;
  for (int my = 0; my < mcu_rows; ++my) {
    for (int mx = 0; mx < mcu_cols; ++mx) {
      for (int i = 0; i < scan_count; ++i) {
        ComponentInfo &c = components[scan_components[i]];
        const std::array<uint16_t, 64> &qtable = qtables[c.quant_table_id];
        const HuffmanTable &dc = dc_tables[c.dc_table_id];
        const HuffmanTable &ac = ac_tables[c.ac_table_id];
        for (int by = 0; by < c.v_sampling; ++by) {
          for (int bx = 0; bx < c.h_sampling; ++bx) {
            std::array<uint8_t, 64> block_samples{};
            if (!DecodeBlock(reader, dc, ac, qtable, c.prev_dc, block_samples)) {
              return false;
            }
            const int sample_x0 = (mx * c.h_sampling + bx) * 8;
            const int sample_y0 = (my * c.v_sampling + by) * 8;
            for (int yy = 0; yy < 8; ++yy) {
              uint8_t *dst = c.samples->data() +
                             static_cast<size_t>(sample_y0 + yy) * c.sample_width + sample_x0;
              const uint8_t *src = block_samples.data() + yy * 8;
              for (int xx = 0; xx < 8; ++xx) {
                dst[xx] = src[xx];
              }
            }
          }
        }
      }
      ++mcu_index;
      if (restart_interval > 0 && mcu_index % restart_interval == 0 &&
          mcu_index < mcu_rows * mcu_cols) {
        reader.ResetForRestart();
        for (int i = 0; i < num_components; ++i) {
          components[i].prev_dc = 0;
        }
      }
    }
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  const size_t out_count =
      static_cast<size_t>(frame_height) * static_cast<size_t>(frame_width) * channels;
  out_pixels.assign(out_count, 0);

  // Per-axis libjpeg-style "fancy" upsampling weights.
  // For subsampling factor 1 the output index maps directly to the
  // input sample (weight (4, 0)). For factor 2 the closer input sample
  // gets weight 3 and the farther one weight 1 (so 2-D weights are
  // 9:3:3:1 / 16 for h2v2 and 3:1 / 4 for h2v1 or h1v2), matching the
  // default behaviour of libjpeg-turbo / Pillow.
  auto fancy_src1 = [](int out_idx, int factor) { return out_idx / factor; };
  auto fancy_src2 = [](int out_idx, int factor, int chroma_dim) {
    if (factor == 1) {
      return out_idx;
    }
    const int src1 = out_idx >> 1;
    const int parity = out_idx & 1;
    int src2 = parity ? src1 + 1 : src1 - 1;
    if (src2 < 0) {
      src2 = 0;
    } else if (src2 >= chroma_dim) {
      src2 = chroma_dim - 1;
    }
    return src2;
  };
  auto fancy_w1 = [](int factor) { return factor == 1 ? 4 : 3; };
  auto fancy_w2 = [](int factor) { return factor == 1 ? 0 : 1; };

  auto sample_component = [&](const ComponentInfo &c, int row, int col) -> uint8_t {
    const int hfac = max_h / c.h_sampling;
    const int vfac = max_v / c.v_sampling;
    const int sy1 = fancy_src1(row, vfac);
    const int sy2 = fancy_src2(row, vfac, c.sample_height);
    const int sx1 = fancy_src1(col, hfac);
    const int sx2 = fancy_src2(col, hfac, c.sample_width);
    const int wy1 = fancy_w1(vfac);
    const int wy2 = fancy_w2(vfac);
    const int wx1 = fancy_w1(hfac);
    const int wx2 = fancy_w2(hfac);
    const int v11 = c.samples->data()[static_cast<size_t>(sy1) * c.sample_width + sx1];
    const int v12 = c.samples->data()[static_cast<size_t>(sy1) * c.sample_width + sx2];
    const int v21 = c.samples->data()[static_cast<size_t>(sy2) * c.sample_width + sx1];
    const int v22 = c.samples->data()[static_cast<size_t>(sy2) * c.sample_width + sx2];
    const int total = (wy1 + wy2) * (wx1 + wx2);
    const int sum = wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;
    return static_cast<uint8_t>((sum + total / 2) / total);
  };

  if (num_components == 1) {
    const ComponentInfo &y = components[0];
    for (int row = 0; row < frame_height; ++row) {
      for (int col = 0; col < frame_width; ++col) {
        const uint8_t yv = sample_component(y, row, col);
        uint8_t *dst =
            out_pixels.data() + (static_cast<size_t>(row) * frame_width + col) * channels;
        if (pixel_format == "Grayscale") {
          dst[0] = yv;
        } else {
          dst[0] = dst[1] = dst[2] = yv;
        }
      }
    }
  } else {
    const ComponentInfo &y = components[0];
    const ComponentInfo &cb = components[1];
    const ComponentInfo &cr = components[2];
    for (int row = 0; row < frame_height; ++row) {
      for (int col = 0; col < frame_width; ++col) {
        const int yy = sample_component(y, row, col);
        const int cb_v = sample_component(cb, row, col);
        const int cr_v = sample_component(cr, row, col);
        // JFIF YCbCr → RGB (ITU-R BT.601, full range).
        const int cb_d = cb_v - 128;
        const int cr_d = cr_v - 128;
        // Integer YCbCr → RGB matching libjpeg's jdcolor.c (SCALEBITS=16).
        constexpr int kScaleBits = 16;
        constexpr int kOneHalf = 1 << (kScaleBits - 1);
        constexpr int kCrToR = 91881;  // FIX(1.40200)
        constexpr int kCbToB = 116130; // FIX(1.77200)
        constexpr int kCbToG = -22554; // -FIX(0.34414)
        constexpr int kCrToG = -46802; // -FIX(0.71414)
        const int r_i = yy + ((kCrToR * cr_d + kOneHalf) >> kScaleBits);
        const int g_i = yy + ((kCbToG * cb_d + kCrToG * cr_d + kOneHalf) >> kScaleBits);
        const int b_i = yy + ((kCbToB * cb_d + kOneHalf) >> kScaleBits);
        auto clamp_u8 = [](int v) { return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); };
        const uint8_t ru = clamp_u8(r_i);
        const uint8_t gu = clamp_u8(g_i);
        const uint8_t bu = clamp_u8(b_i);
        uint8_t *dst =
            out_pixels.data() + (static_cast<size_t>(row) * frame_width + col) * channels;
        if (pixel_format == "RGB") {
          dst[0] = ru;
          dst[1] = gu;
          dst[2] = bu;
        } else if (pixel_format == "BGR") {
          dst[0] = bu;
          dst[1] = gu;
          dst[2] = ru;
        } else {
          // Grayscale: standard luminance formula (ITU-R BT.601).
          dst[0] = static_cast<uint8_t>((299 * ru + 587 * gu + 114 * bu + 500) / 1000);
        }
      }
    }
  }

  out_height = static_cast<int64_t>(frame_height);
  out_width = static_cast<int64_t>(frame_width);
  return true;
}

// ---------------------------------------------------------------------------
// PNG decoder
//
// Decodes a PNG bytestream restricted to the variants used by the upstream
// backend test data: 8-bit non-interlaced color type 0 (Grayscale) or 2
// (RGB), single or multiple ``IDAT`` chunks, deflate compressed. Implements
// DEFLATE (RFC 1951) and zlib (RFC 1950) inline so the kernel remains free
// of external dependencies.
// ---------------------------------------------------------------------------

class DeflateBitReader {
public:
  DeflateBitReader(const uint8_t *data, size_t size) : data_(data), size_(size) {}

  bool error() const { return err_; }

  uint32_t bits(int n) {
    while (nbits_ < n) {
      if (pos_ >= size_) {
        err_ = true;
        return 0;
      }
      buf_ |= static_cast<uint64_t>(data_[pos_++]) << nbits_;
      nbits_ += 8;
    }
    uint32_t v = static_cast<uint32_t>(buf_ & ((static_cast<uint64_t>(1) << n) - 1));
    buf_ >>= n;
    nbits_ -= n;
    return v;
  }

  void align_byte() {
    buf_ = 0;
    nbits_ = 0;
  }

  uint8_t read_byte() {
    align_byte();
    if (pos_ >= size_) {
      err_ = true;
      return 0;
    }
    return data_[pos_++];
  }

private:
  const uint8_t *data_;
  size_t size_;
  size_t pos_ = 0;
  uint64_t buf_ = 0;
  int nbits_ = 0;
  bool err_ = false;
};

struct DeflateHuff {
  // ``counts[len]`` is the number of codes that have length ``len``.
  std::array<int, 16> counts{};
  // Symbols sorted by (length, symbol).
  std::vector<int> symbols;
};

bool BuildDeflateHuff(const int *lengths, int n, DeflateHuff &h) {
  h.counts.fill(0);
  for (int i = 0; i < n; ++i) {
    int l = lengths[i];
    if (l < 0 || l > 15) {
      return false;
    }
    ++h.counts[l];
  }
  h.counts[0] = 0;
  // Kraft inequality check.
  int left = 1;
  for (int len = 1; len <= 15; ++len) {
    left <<= 1;
    left -= h.counts[len];
    if (left < 0) {
      return false;
    }
  }
  std::array<int, 16> offs{};
  for (int len = 1; len < 16; ++len) {
    offs[len] = offs[len - 1] + h.counts[len - 1];
  }
  h.symbols.assign(static_cast<size_t>(n), 0);
  std::array<int, 16> next_off = offs;
  for (int sym = 0; sym < n; ++sym) {
    int l = lengths[sym];
    if (l != 0) {
      h.symbols[static_cast<size_t>(next_off[l]++)] = sym;
    }
  }
  return true;
}

int DecodeDeflate(DeflateBitReader &br, const DeflateHuff &h) {
  int code = 0;
  int first = 0;
  int index = 0;
  for (int len = 1; len <= 15; ++len) {
    int bit = static_cast<int>(br.bits(1));
    if (br.error()) {
      return -1;
    }
    code = (code << 1) | bit;
    int count = h.counts[len];
    if (code - count < first) {
      return h.symbols[static_cast<size_t>(index + (code - first))];
    }
    index += count;
    first = (first + count) << 1;
  }
  return -1;
}

bool Inflate(DeflateBitReader &br, std::vector<uint8_t> &out) {
  static const int kLengthBase[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                      15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                      67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const int kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                       2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const int kDistBase[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                     6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  static const int kCodeLenOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                        11, 4,  12, 3, 13, 2, 14, 1, 15};

  while (true) {
    int bfinal = static_cast<int>(br.bits(1));
    int btype = static_cast<int>(br.bits(2));
    if (br.error()) {
      return false;
    }

    if (btype == 0) {
      uint8_t b0 = br.read_byte();
      uint8_t b1 = br.read_byte();
      uint8_t c0 = br.read_byte();
      uint8_t c1 = br.read_byte();
      if (br.error()) {
        return false;
      }
      uint16_t len = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
      uint16_t nlen = static_cast<uint16_t>(c0) | (static_cast<uint16_t>(c1) << 8);
      if (static_cast<uint16_t>(len ^ 0xFFFF) != nlen) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        uint8_t v = br.read_byte();
        if (br.error()) {
          return false;
        }
        out.push_back(v);
      }
    } else if (btype == 3) {
      return false;
    } else {
      DeflateHuff lit;
      DeflateHuff dist;
      if (btype == 1) {
        std::vector<int> ll(288);
        std::vector<int> dl(30);
        for (int i = 0; i < 144; ++i)
          ll[i] = 8;
        for (int i = 144; i < 256; ++i)
          ll[i] = 9;
        for (int i = 256; i < 280; ++i)
          ll[i] = 7;
        for (int i = 280; i < 288; ++i)
          ll[i] = 8;
        for (int i = 0; i < 30; ++i)
          dl[i] = 5;
        if (!BuildDeflateHuff(ll.data(), 288, lit))
          return false;
        if (!BuildDeflateHuff(dl.data(), 30, dist))
          return false;
      } else {
        int hlit = static_cast<int>(br.bits(5)) + 257;
        int hdist = static_cast<int>(br.bits(5)) + 1;
        int hclen = static_cast<int>(br.bits(4)) + 4;
        if (br.error())
          return false;
        if (hlit > 286 || hdist > 30)
          return false;
        std::vector<int> clens(19, 0);
        for (int i = 0; i < hclen; ++i) {
          clens[kCodeLenOrder[i]] = static_cast<int>(br.bits(3));
        }
        if (br.error())
          return false;
        DeflateHuff clt;
        if (!BuildDeflateHuff(clens.data(), 19, clt))
          return false;
        std::vector<int> codes(static_cast<size_t>(hlit + hdist), 0);
        int i = 0;
        while (i < hlit + hdist) {
          int sym = DecodeDeflate(br, clt);
          if (sym < 0)
            return false;
          if (sym < 16) {
            codes[static_cast<size_t>(i++)] = sym;
          } else if (sym == 16) {
            if (i == 0)
              return false;
            int n = static_cast<int>(br.bits(2)) + 3;
            int prev = codes[static_cast<size_t>(i - 1)];
            while (n-- > 0 && i < hlit + hdist) {
              codes[static_cast<size_t>(i++)] = prev;
            }
          } else if (sym == 17) {
            int n = static_cast<int>(br.bits(3)) + 3;
            while (n-- > 0 && i < hlit + hdist) {
              codes[static_cast<size_t>(i++)] = 0;
            }
          } else {
            int n = static_cast<int>(br.bits(7)) + 11;
            while (n-- > 0 && i < hlit + hdist) {
              codes[static_cast<size_t>(i++)] = 0;
            }
          }
          if (br.error())
            return false;
        }
        if (i != hlit + hdist)
          return false;
        if (!BuildDeflateHuff(codes.data(), hlit, lit))
          return false;
        if (!BuildDeflateHuff(codes.data() + hlit, hdist, dist))
          return false;
      }

      while (true) {
        int sym = DecodeDeflate(br, lit);
        if (sym < 0)
          return false;
        if (sym < 256) {
          out.push_back(static_cast<uint8_t>(sym));
        } else if (sym == 256) {
          break;
        } else {
          int lc = sym - 257;
          if (lc < 0 || lc >= 29)
            return false;
          int length = kLengthBase[lc] + static_cast<int>(br.bits(kLengthExtra[lc]));
          int dsym = DecodeDeflate(br, dist);
          if (dsym < 0 || dsym >= 30)
            return false;
          int distance = kDistBase[dsym] + static_cast<int>(br.bits(kDistExtra[dsym]));
          if (br.error())
            return false;
          if (distance <= 0 || static_cast<size_t>(distance) > out.size())
            return false;
          size_t start = out.size() - static_cast<size_t>(distance);
          for (int k = 0; k < length; ++k) {
            out.push_back(out[start + static_cast<size_t>(k)]);
          }
        }
      }
    }

    if (bfinal)
      break;
  }
  return true;
}

bool TryDecodePng(const uint8_t *data, size_t size, const std::string &pixel_format,
                  int64_t &out_height, int64_t &out_width, RawByteBuffer &out_pixels,
                  RawBufferAllocator *allocator) {
  static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  // 8-byte signature + minimum IHDR (4 len + 4 tag + 13 data + 4 CRC) +
  // minimum IEND (4 + 4 + 0 + 4) = 8 + 25 + 12 = 45.
  if (size < 45) {
    return false;
  }
  for (int i = 0; i < 8; ++i) {
    if (data[i] != kSig[i])
      return false;
  }

  size_t pos = 8;
  int32_t width = 0;
  int32_t height = 0;
  int bit_depth = 0;
  int color_type = 0;
  int compression = 0;
  int filter_method = 0;
  int interlace = 0;
  bool ihdr_seen = false;
  bool iend_seen = false;
  std::vector<uint8_t> idat;

  while (pos + 8 <= size) {
    uint32_t chunk_len = ReadU32BE(data + pos);
    pos += 4;
    char tag[5];
    std::memcpy(tag, data + pos, 4);
    tag[4] = 0;
    pos += 4;
    if (chunk_len > size || pos + static_cast<size_t>(chunk_len) + 4 > size) {
      return false;
    }
    if (std::strcmp(tag, "IHDR") == 0) {
      if (chunk_len != 13 || ihdr_seen)
        return false;
      width = static_cast<int32_t>(ReadU32BE(data + pos));
      height = static_cast<int32_t>(ReadU32BE(data + pos + 4));
      bit_depth = data[pos + 8];
      color_type = data[pos + 9];
      compression = data[pos + 10];
      filter_method = data[pos + 11];
      interlace = data[pos + 12];
      ihdr_seen = true;
    } else if (std::strcmp(tag, "IDAT") == 0) {
      if (!ihdr_seen)
        return false;
      idat.insert(idat.end(), data + pos, data + pos + chunk_len);
    } else if (std::strcmp(tag, "IEND") == 0) {
      iend_seen = true;
      pos += static_cast<size_t>(chunk_len) + 4;
      break;
    }
    pos += static_cast<size_t>(chunk_len) + 4; // chunk data + CRC
  }

  if (!ihdr_seen || !iend_seen)
    return false;
  if (width <= 0 || height <= 0)
    return false;
  if (compression != 0 || filter_method != 0 || interlace != 0)
    return false;
  if (bit_depth != 8)
    return false;
  int src_channels;
  if (color_type == 2) {
    src_channels = 3;
  } else if (color_type == 0) {
    src_channels = 1;
  } else {
    return false;
  }

  // zlib wrapper: 2-byte header, deflate stream, 4-byte Adler-32 (unchecked).
  if (idat.size() < 2)
    return false;
  uint8_t cmf = idat[0];
  uint8_t flg = idat[1];
  if ((cmf & 0x0F) != 8)
    return false; // not deflate
  if ((static_cast<int>(cmf) * 256 + static_cast<int>(flg)) % 31 != 0)
    return false;
  if (flg & 0x20)
    return false; // FDICT (preset dictionary) not supported

  DeflateBitReader br(idat.data() + 2, idat.size() - 2);
  std::vector<uint8_t> raw;
  const size_t row_bytes = static_cast<size_t>(width) * static_cast<size_t>(src_channels);
  const size_t expected = static_cast<size_t>(height) * (1u + row_bytes);
  raw.reserve(expected);
  if (!Inflate(br, raw))
    return false;
  if (raw.size() < expected)
    return false;

  detail::TemporaryTypedBuffer<uint8_t> rows(static_cast<size_t>(height) * row_bytes, allocator,
                                             "PNG rows");
  const int bpp = src_channels;
  for (int r = 0; r < height; ++r) {
    uint8_t filt_type = raw[static_cast<size_t>(r) * (1u + row_bytes)];
    const uint8_t *src = raw.data() + static_cast<size_t>(r) * (1u + row_bytes) + 1;
    uint8_t *dst = rows.data() + static_cast<size_t>(r) * row_bytes;
    const uint8_t *prev = (r == 0) ? nullptr : rows.data() + static_cast<size_t>(r - 1) * row_bytes;
    switch (filt_type) {
    case 0: // None
      std::memcpy(dst, src, row_bytes);
      break;
    case 1: // Sub
      for (size_t x = 0; x < row_bytes; ++x) {
        uint8_t left = (x >= static_cast<size_t>(bpp)) ? dst[x - bpp] : 0;
        dst[x] = static_cast<uint8_t>(src[x] + left);
      }
      break;
    case 2: // Up
      for (size_t x = 0; x < row_bytes; ++x) {
        uint8_t up = prev ? prev[x] : 0;
        dst[x] = static_cast<uint8_t>(src[x] + up);
      }
      break;
    case 3: // Average
      for (size_t x = 0; x < row_bytes; ++x) {
        int left = (x >= static_cast<size_t>(bpp)) ? dst[x - bpp] : 0;
        int up = prev ? prev[x] : 0;
        dst[x] = static_cast<uint8_t>(src[x] + ((left + up) / 2));
      }
      break;
    case 4: // Paeth
      for (size_t x = 0; x < row_bytes; ++x) {
        int a = (x >= static_cast<size_t>(bpp)) ? dst[x - bpp] : 0;
        int b = prev ? prev[x] : 0;
        int c = (prev && x >= static_cast<size_t>(bpp)) ? prev[x - bpp] : 0;
        int p = a + b - c;
        int pa = std::abs(p - a);
        int pb = std::abs(p - b);
        int pc = std::abs(p - c);
        int pr;
        if (pa <= pb && pa <= pc) {
          pr = a;
        } else if (pb <= pc) {
          pr = b;
        } else {
          pr = c;
        }
        dst[x] = static_cast<uint8_t>(src[x] + pr);
      }
      break;
    default:
      return false;
    }
  }

  const int64_t out_channels = ImageDecoder::ChannelCount(pixel_format);
  out_pixels.resize(static_cast<size_t>(height) * static_cast<size_t>(width) *
                    static_cast<size_t>(out_channels));
  for (int y = 0; y < height; ++y) {
    const uint8_t *src_row = rows.data() + static_cast<size_t>(y) * row_bytes;
    uint8_t *dst_row = out_pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(width) *
                                               static_cast<size_t>(out_channels);
    for (int x = 0; x < width; ++x) {
      uint8_t r;
      uint8_t g;
      uint8_t b;
      if (src_channels == 3) {
        r = src_row[x * 3 + 0];
        g = src_row[x * 3 + 1];
        b = src_row[x * 3 + 2];
      } else {
        r = g = b = src_row[x];
      }
      uint8_t *p = dst_row + static_cast<size_t>(x) * static_cast<size_t>(out_channels);
      if (pixel_format == "RGB") {
        p[0] = r;
        p[1] = g;
        p[2] = b;
      } else if (pixel_format == "BGR") {
        p[0] = b;
        p[1] = g;
        p[2] = r;
      } else {
        p[0] = static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000);
      }
    }
  }

  out_height = static_cast<int64_t>(height);
  out_width = static_cast<int64_t>(width);
  return true;
}

// ---------------------------------------------------------------------------
// PNM decoder (Portable AnyMap: PBM/PGM/PPM)
//
// Decodes the Netpbm family of bytestreams without any external dependency:
//   * ``P1`` ASCII bitmap, ``P4`` binary bitmap (1 bit per pixel),
//   * ``P2`` ASCII graymap, ``P5`` binary graymap (8-bit samples),
//   * ``P3`` ASCII pixmap, ``P6`` binary pixmap (8-bit RGB samples).
//
// Only 8-bit (``maxval <= 255``) graymaps and pixmaps are supported; 16-bit
// samples fall through to the empty-matrix path handled by the caller. The
// decoded pixels are converted to the requested ``RGB`` / ``BGR`` /
// ``Grayscale`` channel-last layout.
// ---------------------------------------------------------------------------

inline bool PnmIsWhitespace(uint8_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Advances ``pos`` past any PNM whitespace and ``#`` comments (a comment runs
// to the end of its line).
void PnmSkipWhitespaceAndComments(const uint8_t *data, size_t size, size_t &pos) {
  while (pos < size) {
    const uint8_t c = data[pos];
    if (c == '#') {
      while (pos < size && data[pos] != '\n') {
        ++pos;
      }
    } else if (PnmIsWhitespace(c)) {
      ++pos;
    } else {
      break;
    }
  }
}

// Skips PNM header whitespace and ``#`` comments, then reads one unsigned
// decimal token. Returns false when no decimal token is available.
bool PnmReadUint(const uint8_t *data, size_t size, size_t &pos, long &value) {
  PnmSkipWhitespaceAndComments(data, size, pos);
  if (pos >= size || data[pos] < '0' || data[pos] > '9') {
    return false;
  }
  long v = 0;
  while (pos < size && data[pos] >= '0' && data[pos] <= '9') {
    v = v * 10 + static_cast<long>(data[pos] - '0');
    if (v > 0x7FFFFFFFL) {
      return false;
    }
    ++pos;
  }
  value = v;
  return true;
}

bool TryDecodePnm(const uint8_t *data, size_t size, const std::string &pixel_format,
                  int64_t &out_height, int64_t &out_width, RawByteBuffer &out_pixels,
                  RawBufferAllocator *allocator) {
  if (size < 2 || data[0] != 'P') {
    return false;
  }
  const uint8_t magic = data[1];
  if (magic < '1' || magic > '6') {
    return false;
  }

  size_t pos = 2;
  long width = 0;
  long height = 0;
  long maxval = 1;
  if (!PnmReadUint(data, size, pos, width) || !PnmReadUint(data, size, pos, height)) {
    return false;
  }
  const bool is_bitmap = (magic == '1' || magic == '4');
  if (!is_bitmap) {
    if (!PnmReadUint(data, size, pos, maxval)) {
      return false;
    }
    // Only 8-bit samples are supported; 16-bit graymaps/pixmaps fall back.
    if (maxval < 1 || maxval > 255) {
      return false;
    }
  }
  if (width <= 0 || height <= 0) {
    return false;
  }

  const int src_channels = (magic == '3' || magic == '6') ? 3 : 1;
  const size_t w = static_cast<size_t>(width);
  const size_t h = static_cast<size_t>(height);
  if (w > ((std::numeric_limits<size_t>::max)() / h)) {
    return false;
  }
  const size_t pixel_count = w * h;
  if (pixel_count > ((std::numeric_limits<size_t>::max)() / static_cast<size_t>(src_channels))) {
    return false;
  }
  const size_t sample_count = pixel_count * static_cast<size_t>(src_channels);

  detail::TemporaryTypedBuffer<uint8_t> src(sample_count, allocator, "PNM src");
  const bool is_binary = (magic == '4' || magic == '5' || magic == '6');
  if (is_binary) {
    // Exactly one whitespace byte separates the header from the binary data.
    if (pos >= size || !PnmIsWhitespace(data[pos])) {
      return false;
    }
    ++pos;
    if (magic == '4') {
      // Packed bitmap: 8 pixels per byte, MSB first, rows padded to a byte.
      const size_t row_bytes = (w + 7u) / 8u;
      if (row_bytes > ((std::numeric_limits<size_t>::max)() / h) || pos + row_bytes * h > size) {
        return false;
      }
      for (size_t y = 0; y < h; ++y) {
        const uint8_t *row = data + pos + y * row_bytes;
        for (size_t x = 0; x < w; ++x) {
          const uint8_t bit = (row[x >> 3] >> (7u - (x & 7u))) & 1u;
          // In PBM a set bit means black; map to 0 (black) / 255 (white).
          src.data()[y * w + x] = bit ? 0u : 255u;
        }
      }
    } else {
      if (pos + sample_count > size) {
        return false;
      }
      std::memcpy(src.data(), data + pos, sample_count);
    }
  } else if (magic == '1') {
    // ASCII bitmap: each pixel is a single ``0`` or ``1`` digit, optionally
    // separated by whitespace or comments.
    size_t idx = 0;
    while (idx < pixel_count) {
      PnmSkipWhitespaceAndComments(data, size, pos);
      if (pos >= size || (data[pos] != '0' && data[pos] != '1')) {
        return false;
      }
      src.data()[idx++] = (data[pos] == '1') ? 0u : 255u;
      ++pos;
    }
  } else {
    // ASCII graymap/pixmap: whitespace-separated decimal samples.
    for (size_t i = 0; i < sample_count; ++i) {
      long v = 0;
      if (!PnmReadUint(data, size, pos, v)) {
        return false;
      }
      if (v > maxval) {
        v = maxval;
      }
      src.data()[i] = static_cast<uint8_t>(v);
    }
  }

  // Scale 8-bit samples from ``[0, maxval]`` to ``[0, 255]`` when needed.
  if (!is_bitmap && maxval != 255) {
    for (size_t i = 0; i < sample_count; ++i) {
      src.data()[i] =
          static_cast<uint8_t>((static_cast<int>(src.data()[i]) * 255 + maxval / 2) / maxval);
    }
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  out_pixels.resize(pixel_count * static_cast<size_t>(channels));
  for (size_t i = 0; i < pixel_count; ++i) {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    if (src_channels == 3) {
      r = src.data()[i * 3u + 0u];
      g = src.data()[i * 3u + 1u];
      b = src.data()[i * 3u + 2u];
    } else {
      r = g = b = src.data()[i];
    }
    if (pixel_format == "RGB") {
      out_pixels[i * 3u + 0u] = r;
      out_pixels[i * 3u + 1u] = g;
      out_pixels[i * 3u + 2u] = b;
    } else if (pixel_format == "BGR") {
      out_pixels[i * 3u + 0u] = b;
      out_pixels[i * 3u + 1u] = g;
      out_pixels[i * 3u + 2u] = r;
    } else {
      out_pixels[i] = static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000);
    }
  }

  out_height = static_cast<int64_t>(height);
  out_width = static_cast<int64_t>(width);
  return true;
}

} // namespace

int64_t ImageDecoder::ChannelCount(const std::string &pixel_format) {
  if (pixel_format == "RGB" || pixel_format == "BGR") {
    return 3;
  }
  if (pixel_format == "Grayscale") {
    return 1;
  }
  EXT_THROW_INVALID("kernel::ImageDecoder: unsupported pixel_format \"", pixel_format,
                    "\" (expected \"RGB\", \"BGR\" or \"Grayscale\").");
}

Tensor ImageDecoder::operator()(const Tensor &encoded_stream, const std::string &pixel_format,
                                RuntimeContext *rt) const {
  CheckImageDecoderInput(encoded_stream);
  const int64_t channels = ChannelCount(pixel_format);

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  RawByteBuffer pixels;
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  if (TryDecodePng(raw, raw_size, pixel_format, height, width, pixels, allocator) ||
      TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels) ||
      TryDecodeJpeg(raw, raw_size, pixel_format, height, width, pixels, allocator) ||
      TryDecodePnm(raw, raw_size, pixel_format, height, width, pixels, allocator)) {
    if (allocator == nullptr) {
      return Tensor::FromRawBytes("", DataType::UINT8, {height, width, channels},
                                  std::move(pixels));
    }
    Tensor output =
        MakeOutputTensor(DataType::UINT8, {height, width, channels}, pixels.size(), allocator);
    std::memcpy(output.mutable_bytes(), pixels.data(), pixels.size());
    return output;
  }

  // Per the ONNX schema, fall back to an empty ``(0, 0, C)`` matrix when
  // the bytestream cannot be decoded.
  return Tensor::FromUint8("", {0, 0, channels}, {}, allocator);
}

void ImageDecoder::operator()(const Tensor &encoded_stream, const std::string &pixel_format,
                              Tensor &output) const {
  CheckImageDecoderInput(encoded_stream);
  const int64_t channels = ChannelCount(pixel_format);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::ImageDecoder preallocated output must be a UINT8 tensor.");
  EXT_ENFORCE_INVALID(output.shape.size() == 3u,
                      "kernel::ImageDecoder preallocated output must be a 3-D tensor "
                      "with channel-last layout (H, W, C).");
  EXT_ENFORCE_INVALID(output.shape[2] == channels,
                      "kernel::ImageDecoder preallocated output channel count does not "
                      "match ``pixel_format``.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.size_bytes()) == output.element_count(),
                      "kernel::ImageDecoder preallocated output data size does not match "
                      "its shape.");

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  RawByteBuffer pixels;
  if (TryDecodePng(raw, raw_size, pixel_format, height, width, pixels, nullptr) ||
      TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels) ||
      TryDecodeJpeg(raw, raw_size, pixel_format, height, width, pixels, nullptr) ||
      TryDecodePnm(raw, raw_size, pixel_format, height, width, pixels, nullptr)) {
    EXT_ENFORCE_INVALID(output.shape[0] == height && output.shape[1] == width,
                        "kernel::ImageDecoder preallocated output shape does not match the decoded "
                        "image dimensions.");
    if (output.has_allocation()) {
      std::memcpy(output.mutable_bytes(), pixels.data(), pixels.size());
    } else {
      output.data = RawBuffer(std::move(pixels));
    }
    return;
  }

  // Fallback empty-matrix path: the caller must provide a ``(0, 0, C)``
  // tensor since the kernel cannot decode the bytestream.
  EXT_ENFORCE_INVALID(output.shape[0] == 0 && output.shape[1] == 0,
                      "kernel::ImageDecoder preallocated output must be (0, 0, C) for the "
                      "empty-matrix fallback used by the lightweight reference kernel.");
}

void ImageDecoder::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &encoded_stream = GetInput(node, 0, rt.tensors());
  const std::string pixel_format = GetAttributeStringOrDefault(node, "pixel_format", "RGB");
  onnx_kernels::kernel::ImageDecoder image_decoder_kernel(rt.kernel_ctx());
  SetOutput(node, 0, image_decoder_kernel(encoded_stream, pixel_format, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
