// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/image/include_image_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

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
                  int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels) {
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
  std::vector<uint8_t> samples;
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
  std::array<double, 64> tmp{};
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      double sum = 0.0;
      for (int u = 0; u < 8; ++u) {
        const double cu = u == 0 ? inv_sqrt2 : 1.0;
        sum += cu * in[u * 8 + x] * std::cos((2 * y + 1) * u * M_PI / 16.0);
      }
      tmp[y * 8 + x] = sum;
    }
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      double sum = 0.0;
      for (int v = 0; v < 8; ++v) {
        const double cv = v == 0 ? inv_sqrt2 : 1.0;
        sum += cv * tmp[y * 8 + v] * std::cos((2 * x + 1) * v * M_PI / 16.0);
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
                   int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels) {
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
    c.samples.assign(static_cast<size_t>(c.sample_width) * c.sample_height, 0);
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
              uint8_t *dst = c.samples.data() +
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
    const int v11 = c.samples[static_cast<size_t>(sy1) * c.sample_width + sx1];
    const int v12 = c.samples[static_cast<size_t>(sy1) * c.sample_width + sx2];
    const int v21 = c.samples[static_cast<size_t>(sy2) * c.sample_width + sx1];
    const int v22 = c.samples[static_cast<size_t>(sy2) * c.sample_width + sx2];
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

} // namespace

int64_t ImageDecoder::ChannelCount(const std::string &pixel_format) {
  if (pixel_format == "RGB" || pixel_format == "BGR") {
    return 3;
  }
  if (pixel_format == "Grayscale") {
    return 1;
  }
  throw std::invalid_argument("kernel::ImageDecoder: unsupported pixel_format \"" + pixel_format +
                              "\" (expected \"RGB\", \"BGR\" or \"Grayscale\").");
}

Tensor ImageDecoder::operator()(const Tensor &encoded_stream,
                                const std::string &pixel_format) const {
  CheckImageDecoderInput(encoded_stream);
  const int64_t channels = ChannelCount(pixel_format);

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  std::vector<uint8_t> pixels;
  if (TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels) ||
      TryDecodeJpeg(raw, raw_size, pixel_format, height, width, pixels)) {
    return Tensor::FromUint8("", {height, width, channels}, std::move(pixels));
  }

  // Per the ONNX schema, fall back to an empty ``(0, 0, C)`` matrix when
  // the bytestream cannot be decoded.
  return Tensor::FromUint8("", {0, 0, channels}, {});
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
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) == output.element_count(),
                      "kernel::ImageDecoder preallocated output data size does not match "
                      "its shape.");

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  std::vector<uint8_t> pixels;
  if (TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels) ||
      TryDecodeJpeg(raw, raw_size, pixel_format, height, width, pixels)) {
    EXT_ENFORCE_INVALID(output.shape[0] == height && output.shape[1] == width,
                        "kernel::ImageDecoder preallocated output shape does not match the decoded "
                        "image dimensions.");
    output.data = std::move(pixels);
    return;
  }

  // Fallback empty-matrix path: the caller must provide a ``(0, 0, C)``
  // tensor since the kernel cannot decode the bytestream.
  EXT_ENFORCE_INVALID(output.shape[0] == 0 && output.shape[1] == 0,
                      "kernel::ImageDecoder preallocated output must be (0, 0, C) for the "
                      "empty-matrix fallback used by the lightweight reference kernel.");
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
