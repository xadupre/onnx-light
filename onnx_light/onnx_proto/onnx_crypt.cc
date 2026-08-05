#include "onnx_crypt.h"

#ifdef ONNX_LIGHT_HAS_OPENSSL

#include "stream.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <vector>

// ChaCha20-Poly1305 is only available when OpenSSL is built with ChaCha support.
// Some OpenSSL builds define OPENSSL_NO_CHACHA, in which case EVP_chacha20_poly1305()
// is not declared. Gate the ChaCha20-Poly1305 code paths on its availability.
#ifndef OPENSSL_NO_CHACHA
#define ONNX_LIGHT_HAS_CHACHA20 1
#else
#define ONNX_LIGHT_HAS_CHACHA20 0
#endif

namespace ONNX_LIGHT_NAMESPACE {

namespace {

// File format constants.
static constexpr std::array<char, 8> MAGIC_AES{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '1'};
static constexpr std::array<char, 8> MAGIC_CHACHA20{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '2'};
static constexpr int SALT_LEN = 16;
static constexpr int IV_LEN = 16; // AES-CBC IV size
static constexpr int NONCE_LEN = 12;
static constexpr int TAG_LEN = 16;
static constexpr int KEY_LEN = 32; // 256-bit key
static constexpr int PBKDF2_ITER = 100000;

// Collects the most recent OpenSSL error string.
std::string openssl_last_error() {
  unsigned long err_code = ERR_get_error();
  if (err_code == 0)
    return "(no OpenSSL error)";
  char buf[256];
  ERR_error_string_n(err_code, buf, sizeof(buf));
  return std::string(buf);
}

// Derives a 32-byte AES key from *key* and *salt* using PBKDF2-HMAC-SHA256.
std::array<uint8_t, KEY_LEN> derive_key(const std::string &key, const uint8_t (&salt)[SALT_LEN]) {
  std::array<uint8_t, KEY_LEN> out{};
  if (PKCS5_PBKDF2_HMAC(key.data(), static_cast<int>(key.size()), salt, SALT_LEN, PBKDF2_ITER,
                        EVP_sha256(), KEY_LEN, out.data()) != 1) {
    throw std::runtime_error("PBKDF2 key derivation failed: " + openssl_last_error());
  }
  return out;
}

// Encrypts *plain* bytes using AES-256-CBC and returns the ONNXCRY1 blob.
std::string encrypt_to_blob_aes(const std::string &plain, const std::string &key) {
  uint8_t salt[SALT_LEN];
  uint8_t iv[IV_LEN];
  if (RAND_bytes(salt, SALT_LEN) != 1 || RAND_bytes(iv, IV_LEN) != 1) {
    throw std::runtime_error("RAND_bytes failed: " + openssl_last_error());
  }

  auto aes_key = derive_key(key, salt);

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());

  if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, aes_key.data(), iv) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptInit_ex failed: " + openssl_last_error());
  }

  const int block_size = EVP_CIPHER_CTX_block_size(ctx);
  std::vector<uint8_t> ciphertext(plain.size() + static_cast<size_t>(block_size));
  int out_len = 0;
  int final_len = 0;

  if (EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len,
                        reinterpret_cast<const uint8_t *>(plain.data()),
                        static_cast<int>(plain.size())) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptUpdate failed: " + openssl_last_error());
  }
  if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptFinal_ex failed: " + openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);
  const size_t cipher_len = static_cast<size_t>(out_len + final_len);

  // Assemble: magic + salt + iv + ciphertext.
  std::string blob;
  blob.reserve(MAGIC_AES.size() + SALT_LEN + IV_LEN + cipher_len);
  blob.append(MAGIC_AES.data(), MAGIC_AES.size());
  blob.append(reinterpret_cast<const char *>(salt), SALT_LEN);
  blob.append(reinterpret_cast<const char *>(iv), IV_LEN);
  blob.append(reinterpret_cast<const char *>(ciphertext.data()), cipher_len);
  return blob;
}

// Encrypts *plain* bytes using ChaCha20-Poly1305 and returns the ONNXCRY2 blob.
std::string encrypt_to_blob_chacha20(const std::string &plain, const std::string &key) {
#if !ONNX_LIGHT_HAS_CHACHA20
  (void)plain;
  (void)key;
  throw std::runtime_error(
      "ChaCha20-Poly1305 is not supported by this OpenSSL build (OPENSSL_NO_CHACHA).");
#else
  uint8_t salt[SALT_LEN];
  uint8_t nonce[NONCE_LEN];
  if (RAND_bytes(salt, SALT_LEN) != 1 || RAND_bytes(nonce, NONCE_LEN) != 1) {
    throw std::runtime_error("RAND_bytes failed: " + openssl_last_error());
  }

  auto encryption_key = derive_key(key, salt);

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());

  if (EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), nullptr, nullptr, nullptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptInit_ex(ChaCha20-Poly1305) failed: " +
                             openssl_last_error());
  }
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, NONCE_LEN, nullptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CTRL_AEAD_SET_IVLEN failed: " + openssl_last_error());
  }
  if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, encryption_key.data(), nonce) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptInit_ex(ChaCha20 key/nonce) failed: " +
                             openssl_last_error());
  }

  std::vector<uint8_t> ciphertext(plain.size());
  int out_len = 0;
  if (EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len,
                        reinterpret_cast<const uint8_t *>(plain.data()),
                        static_cast<int>(plain.size())) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptUpdate(ChaCha20) failed: " + openssl_last_error());
  }
  int final_len = 0;
  if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptFinal_ex(ChaCha20) failed: " + openssl_last_error());
  }
  uint8_t tag[TAG_LEN];
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, TAG_LEN, tag) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CTRL_AEAD_GET_TAG failed: " + openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);

  const size_t cipher_len = static_cast<size_t>(out_len + final_len);
  ciphertext.resize(cipher_len);

  std::string blob;
  blob.reserve(MAGIC_CHACHA20.size() + SALT_LEN + NONCE_LEN + TAG_LEN + cipher_len);
  blob.append(MAGIC_CHACHA20.data(), MAGIC_CHACHA20.size());
  blob.append(reinterpret_cast<const char *>(salt), SALT_LEN);
  blob.append(reinterpret_cast<const char *>(nonce), NONCE_LEN);
  blob.append(reinterpret_cast<const char *>(tag), TAG_LEN);
  blob.append(reinterpret_cast<const char *>(ciphertext.data()), cipher_len);
  return blob;
#endif
}

std::string encrypt_to_blob(const std::string &plain, const std::string &key,
                            const std::string &encryption) {
  if (encryption == "AES-256-CBC")
    return encrypt_to_blob_aes(plain, key);
  if (encryption == "ChaCha20-Poly1305")
    return encrypt_to_blob_chacha20(plain, key);
  throw std::runtime_error("Unsupported encryption algorithm '" + encryption +
                           "'. Supported values: AES-256-CBC, ChaCha20-Poly1305.");
}

std::vector<uint8_t> decrypt_from_blob_aes(const uint8_t *data, size_t data_len,
                                           const std::string &key) {
  const size_t header_size = MAGIC_AES.size() + SALT_LEN + IV_LEN;
  if (data_len < header_size)
    throw std::runtime_error("Buffer too small to be a valid ONNXCRY1 payload.");

  if (std::memcmp(data, MAGIC_AES.data(), MAGIC_AES.size()) != 0)
    throw std::runtime_error("Bad magic bytes – not an ONNXCRY1 encrypted payload.");

  const uint8_t *salt_ptr = data + MAGIC_AES.size();
  const uint8_t *iv_ptr = salt_ptr + SALT_LEN;
  const uint8_t *ciphertext = iv_ptr + IV_LEN;
  const size_t cipher_len = data_len - header_size;

  uint8_t salt_arr[SALT_LEN];
  std::memcpy(salt_arr, salt_ptr, SALT_LEN);
  auto aes_key = derive_key(key, salt_arr);

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());

  if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, aes_key.data(), iv_ptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptInit_ex failed: " + openssl_last_error());
  }

  std::vector<uint8_t> plaintext(cipher_len + static_cast<size_t>(EVP_CIPHER_CTX_block_size(ctx)));
  int out_len = 0;
  int final_len = 0;

  if (EVP_DecryptUpdate(ctx, plaintext.data(), &out_len, ciphertext,
                        static_cast<int>(cipher_len)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptUpdate failed (wrong key?): " + openssl_last_error());
  }
  if (EVP_DecryptFinal_ex(ctx, plaintext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptFinal_ex failed (wrong key or corrupt data): " +
                             openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);
  plaintext.resize(static_cast<size_t>(out_len + final_len));
  return plaintext;
}

std::vector<uint8_t> decrypt_from_blob_chacha20(const uint8_t *data, size_t data_len,
                                                const std::string &key) {
#if !ONNX_LIGHT_HAS_CHACHA20
  (void)data;
  (void)data_len;
  (void)key;
  throw std::runtime_error(
      "ChaCha20-Poly1305 is not supported by this OpenSSL build (OPENSSL_NO_CHACHA).");
#else
  const size_t header_size = MAGIC_CHACHA20.size() + SALT_LEN + NONCE_LEN + TAG_LEN;
  if (data_len < header_size)
    throw std::runtime_error("Buffer too small to be a valid ONNXCRY2 payload.");

  if (std::memcmp(data, MAGIC_CHACHA20.data(), MAGIC_CHACHA20.size()) != 0)
    throw std::runtime_error("Bad magic bytes – not an ONNXCRY2 encrypted payload.");

  const uint8_t *salt_ptr = data + MAGIC_CHACHA20.size();
  const uint8_t *nonce_ptr = salt_ptr + SALT_LEN;
  const uint8_t *tag_ptr = nonce_ptr + NONCE_LEN;
  const uint8_t *ciphertext = tag_ptr + TAG_LEN;
  const size_t cipher_len = data_len - header_size;

  uint8_t salt_arr[SALT_LEN];
  std::memcpy(salt_arr, salt_ptr, SALT_LEN);
  auto encryption_key = derive_key(key, salt_arr);

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());
  if (EVP_DecryptInit_ex(ctx, EVP_chacha20_poly1305(), nullptr, nullptr, nullptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptInit_ex(ChaCha20-Poly1305) failed: " +
                             openssl_last_error());
  }
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, NONCE_LEN, nullptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CTRL_AEAD_SET_IVLEN failed: " + openssl_last_error());
  }
  if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, encryption_key.data(), nonce_ptr) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptInit_ex(ChaCha20 key/nonce) failed: " +
                             openssl_last_error());
  }

  std::vector<uint8_t> plaintext(cipher_len);
  int out_len = 0;
  if (EVP_DecryptUpdate(ctx, plaintext.data(), &out_len, ciphertext,
                        static_cast<int>(cipher_len)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptUpdate(ChaCha20) failed: " + openssl_last_error());
  }
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, TAG_LEN, const_cast<uint8_t *>(tag_ptr)) !=
      1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CTRL_AEAD_SET_TAG failed: " + openssl_last_error());
  }
  int final_len = 0;
  if (EVP_DecryptFinal_ex(ctx, plaintext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptFinal_ex(ChaCha20) failed (wrong key or corrupt data): " +
                             openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);

  plaintext.resize(static_cast<size_t>(out_len + final_len));
  return plaintext;
#endif
}

// Decrypts an ONNXCRY1 or ONNXCRY2 *blob* and returns the plaintext bytes.
std::vector<uint8_t> decrypt_from_blob(const uint8_t *data, size_t data_len,
                                       const std::string &key) {
  if (data_len < MAGIC_AES.size())
    throw std::runtime_error("Buffer too small to be a valid encrypted payload.");
  if (std::memcmp(data, MAGIC_AES.data(), MAGIC_AES.size()) == 0)
    return decrypt_from_blob_aes(data, data_len, key);
  if (std::memcmp(data, MAGIC_CHACHA20.data(), MAGIC_CHACHA20.size()) == 0)
    return decrypt_from_blob_chacha20(data, data_len, key);
  throw std::runtime_error("Bad magic bytes – not an ONNXCRY1/ONNXCRY2 encrypted payload.");
}

} // namespace

void SaveEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const SerializeOptions &opts, const std::string &encryption) {
  std::string plain;
  SerializeOptions mutable_opts = opts;
  model.SerializeToString(plain, mutable_opts);

  const std::string blob = encrypt_to_blob(plain, key, encryption);

  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open())
    throw std::runtime_error("Cannot open file for writing: " + file_path);
  ofs.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  if (!ofs)
    throw std::runtime_error("Write error for file: " + file_path);
}

void SaveEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const std::string &encryption) {
  SaveEncryptedModel(model, file_path, key, SerializeOptions{}, encryption);
}

void LoadEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const ParseOptions &opts) {
  std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open encrypted file: " + file_path);

  const std::streamsize file_size = ifs.tellg();
  if (file_size < 0)
    throw std::runtime_error("Cannot determine size of encrypted file: " + file_path);
  ifs.seekg(0, std::ios::beg);
  std::string file_buf(static_cast<size_t>(file_size), '\0');
  ifs.read(file_buf.data(), file_size);
  if (!ifs)
    throw std::runtime_error("Read error for file: " + file_path);

  LoadEncryptedModelFromString(model, file_buf, key, opts);
}

std::string SaveEncryptedModelToString(ModelProto &model, const std::string &key,
                                       const SerializeOptions &opts,
                                       const std::string &encryption) {
  std::string plain;
  SerializeOptions mutable_opts = opts;
  model.SerializeToString(plain, mutable_opts);
  return encrypt_to_blob(plain, key, encryption);
}

std::string SaveEncryptedModelToString(ModelProto &model, const std::string &key,
                                       const std::string &encryption) {
  return SaveEncryptedModelToString(model, key, SerializeOptions{}, encryption);
}

void LoadEncryptedModelFromString(ModelProto &model, const std::string &encrypted_data,
                                  const std::string &key, const ParseOptions &opts) {
  auto plaintext = decrypt_from_blob(reinterpret_cast<const uint8_t *>(encrypted_data.data()),
                                     encrypted_data.size(), key);
  ParseOptions mutable_opts = opts;
  utils::StringStream stream(plaintext.data(), static_cast<int64_t>(plaintext.size()));
  if (mutable_opts.is_parallel())
    stream.StartThreadPool(mutable_opts.num_threads);
  // AES-CBC has no MAC, so a wrong key still produces valid-looking plaintext
  // about 1/256 of the time (whenever the PKCS#7 padding happens to be valid).
  // The downstream proto parser then operates on garbage bytes and may throw a
  // std::runtime_error whose message embeds those raw bytes — which nanobind
  // cannot translate to a Python ``str`` (UTF-8), yielding an unrelated
  // ``UnicodeDecodeError``. Re-raise here with a sanitized ASCII-only message
  // so callers always observe a single, predictable exception type while still
  // preserving the underlying parser error (which may also flag genuinely
  // corrupt data even with the right key).
  try {
    model.ParseFromStream(stream, mutable_opts);
    if (mutable_opts.is_parallel())
      stream.WaitForDelayedBlock();
  } catch (const std::exception &ex) {
    if (mutable_opts.is_parallel())
      stream.WaitForDelayedBlock();
    std::string sanitized;
    sanitized.reserve(std::strlen(ex.what()));
    for (const char *p = ex.what(); *p != '\0'; ++p) {
      const unsigned char c = static_cast<unsigned char>(*p);
      if (c >= 0x20 && c < 0x7F) {
        sanitized.push_back(static_cast<char>(c));
      } else {
        char hex[5];
        std::snprintf(hex, sizeof(hex), "\\x%02x", c);
        sanitized.append(hex);
      }
    }
    throw std::runtime_error(
        "Failed to parse decrypted ONNXCRY1 payload (wrong key or corrupt data): " + sanitized);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE

#endif // ONNX_LIGHT_HAS_OPENSSL
