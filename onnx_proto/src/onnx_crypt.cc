#include "onnx_crypt.h"

#ifdef ONNX_LIGHT_HAS_OPENSSL

#include "external_data_internal.h"
#include "stream.h"
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <system_error>
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

using utils::detail::kExternalDataDigestKey;
using utils::detail::kExternalDataEncryptionFormat;
using utils::detail::kExternalDataEncryptionKey;
using utils::detail::kExternalDataLengthKey;
using utils::detail::kExternalDataLocationKey;
using utils::detail::kExternalDataOffsetKey;

// File format constants.
static constexpr std::array<char, 8> MAGIC_AES{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '1'};
static constexpr std::array<char, 8> MAGIC_CHACHA20{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '2'};
static constexpr std::array<char, 8> MAGIC_GCM{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '3'};
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
  if (key.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("Encryption key exceeds INT_MAX bytes.");
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

// Encrypts *plain* bytes using AES-256-GCM and authenticates the magic, salt, and nonce.
std::string encrypt_to_blob_gcm(const std::string &plain, const std::string &key) {
  if (plain.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("ONNXCRY3 payload exceeds INT_MAX bytes.");

  uint8_t salt[SALT_LEN];
  uint8_t nonce[NONCE_LEN];
  if (RAND_bytes(salt, SALT_LEN) != 1 || RAND_bytes(nonce, NONCE_LEN) != 1)
    throw std::runtime_error("RAND_bytes failed: " + openssl_last_error());
  auto encryption_key = derive_key(key, salt);

  std::string blob;
  blob.reserve(MAGIC_GCM.size() + SALT_LEN + NONCE_LEN + TAG_LEN + plain.size());
  blob.append(MAGIC_GCM.data(), MAGIC_GCM.size());
  blob.append(reinterpret_cast<const char *>(salt), SALT_LEN);
  blob.append(reinterpret_cast<const char *>(nonce), NONCE_LEN);

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());
  if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, encryption_key.data(), nonce) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptInit_ex(AES-256-GCM) failed: " + openssl_last_error());
  }
  int aad_len = 0;
  if (EVP_EncryptUpdate(ctx, nullptr, &aad_len, reinterpret_cast<const uint8_t *>(blob.data()),
                        static_cast<int>(blob.size())) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_EncryptUpdate(AES-256-GCM AAD) failed: " + openssl_last_error());
  }
  std::vector<uint8_t> ciphertext(plain.size() + EVP_MAX_BLOCK_LENGTH);
  int out_len = 0;
  int final_len = 0;
  if (EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len,
                        reinterpret_cast<const uint8_t *>(plain.data()),
                        static_cast<int>(plain.size())) != 1 ||
      EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_Encrypt(AES-256-GCM) failed: " + openssl_last_error());
  }
  uint8_t tag[TAG_LEN];
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, TAG_LEN, tag) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CTRL_GCM_GET_TAG failed: " + openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);
  blob.append(reinterpret_cast<const char *>(tag), TAG_LEN);
  blob.append(reinterpret_cast<const char *>(ciphertext.data()),
              static_cast<size_t>(out_len + final_len));
  return blob;
}

std::string encrypt_to_blob(const std::string &plain, const std::string &key,
                            const std::string &encryption) {
  if (encryption == "AES-256-CBC")
    return encrypt_to_blob_aes(plain, key);
  if (encryption == "ChaCha20-Poly1305")
    return encrypt_to_blob_chacha20(plain, key);
  if (encryption == "AES-256-GCM")
    return encrypt_to_blob_gcm(plain, key);
  throw std::runtime_error("Unsupported encryption algorithm '" + encryption +
                           "'. Supported values: AES-256-CBC, ChaCha20-Poly1305, AES-256-GCM.");
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

std::vector<uint8_t> decrypt_from_blob_gcm(const uint8_t *data, size_t data_len,
                                           const std::string &key) {
  const size_t aad_size = MAGIC_GCM.size() + SALT_LEN + NONCE_LEN;
  const size_t header_size = aad_size + TAG_LEN;
  if (data_len < header_size)
    throw std::runtime_error("Buffer too small for an ONNXCRY3 payload.");
  const size_t cipher_len = data_len - header_size;
  if (cipher_len > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("ONNXCRY3 ciphertext exceeds INT_MAX bytes.");

  uint8_t salt[SALT_LEN];
  std::memcpy(salt, data + MAGIC_GCM.size(), SALT_LEN);
  auto encryption_key = derive_key(key, salt);
  const uint8_t *nonce = data + MAGIC_GCM.size() + SALT_LEN;
  const uint8_t *tag = data + aad_size;
  const uint8_t *ciphertext = data + header_size;
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());
  if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, encryption_key.data(), nonce) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptInit_ex(AES-256-GCM) failed: " + openssl_last_error());
  }
  int aad_len = 0;
  if (EVP_DecryptUpdate(ctx, nullptr, &aad_len, data, static_cast<int>(aad_size)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptUpdate(AES-256-GCM AAD) failed: " + openssl_last_error());
  }
  std::vector<uint8_t> plaintext(cipher_len + EVP_MAX_BLOCK_LENGTH);
  int out_len = 0;
  if (EVP_DecryptUpdate(ctx, plaintext.data(), &out_len, ciphertext,
                        static_cast<int>(cipher_len)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_DecryptUpdate(AES-256-GCM ciphertext) failed: " +
                             openssl_last_error());
  }
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, TAG_LEN, const_cast<uint8_t *>(tag)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("EVP_CIPHER_CTX_ctrl(EVP_CTRL_GCM_SET_TAG) failed: " +
                             openssl_last_error());
  }
  int final_len = 0;
  if (EVP_DecryptFinal_ex(ctx, plaintext.data() + out_len, &final_len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("ONNXCRY3 authentication failed (wrong key or corrupt data).");
  }
  EVP_CIPHER_CTX_free(ctx);
  plaintext.resize(static_cast<size_t>(out_len + final_len));
  return plaintext;
}

// Decrypts an ONNXCRY1, ONNXCRY2, or ONNXCRY3 *blob* and returns the plaintext bytes.
std::vector<uint8_t> decrypt_from_blob(const uint8_t *data, size_t data_len,
                                       const std::string &key) {
  if (data_len < MAGIC_AES.size())
    throw std::runtime_error("Buffer too small to be a valid encrypted payload.");
  if (std::memcmp(data, MAGIC_AES.data(), MAGIC_AES.size()) == 0)
    return decrypt_from_blob_aes(data, data_len, key);
  if (std::memcmp(data, MAGIC_CHACHA20.data(), MAGIC_CHACHA20.size()) == 0)
    return decrypt_from_blob_chacha20(data, data_len, key);
  if (std::memcmp(data, MAGIC_GCM.data(), MAGIC_GCM.size()) == 0)
    return decrypt_from_blob_gcm(data, data_len, key);
  throw std::runtime_error(
      "Bad magic bytes – not an ONNXCRY1/ONNXCRY2/ONNXCRY3 encrypted payload.");
}

constexpr size_t GCM_HEADER_SIZE = MAGIC_GCM.size() + SALT_LEN + NONCE_LEN + TAG_LEN;
static_assert(GCM_HEADER_SIZE == 52);

void validate_external_limits(const std::string &key, size_t limit) {
  if (key.empty() || key.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "Encrypted external data requires a nonempty key of at most INT_MAX bytes.");
  if (limit <= GCM_HEADER_SIZE ||
      limit - GCM_HEADER_SIZE > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "Encrypted external file size must be between 53 and INT_MAX + 52 bytes.");
}

std::string hex_bytes(const uint8_t *bytes, size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(size * 2);
  for (size_t i = 0; i < size; ++i) {
    result.push_back(digits[bytes[i] >> 4]);
    result.push_back(digits[bytes[i] & 15]);
  }
  return result;
}

std::string encrypted_digest(const std::string &blob) {
  uint8_t digest[EVP_MAX_MD_SIZE];
  unsigned int size = 0;
  if (EVP_Digest(blob.data(), blob.size(), digest, &size, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("Encrypted external data SHA-256 failed: " + openssl_last_error());
  return hex_bytes(digest, size);
}

void write_encrypted_file(const std::filesystem::path &path, const std::string &blob) {
  const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  auto file = utils::detail::OpenExternalDataFile(
      utils::detail::external_path_to_utf8(parent),
      utils::detail::external_path_to_utf8(path.filename()), true);
  utils::FileWriteStream stream(utils::detail::external_path_to_utf8(path), std::move(file));
  stream.write_raw_bytes(reinterpret_cast<const uint8_t *>(blob.data()), blob.size());
}

void add_external_entry(TensorProto &tensor, const std::string &key, const std::string &value) {
  auto *entry = tensor.add_external_data();
  entry->set_key(key);
  entry->set_value(value);
}

bool parse_external_integer(const std::string &value, int64_t &result) {
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

std::string external_tensor_context(const TensorProto &tensor, const std::string &location) {
  return "tensor '" + std::string(tensor.name()) + "', location '" + location + "'";
}

void reject_unsupported_external_data(const GraphProto &graph, bool codec = false);
void reject_unsupported_external_data(const FunctionProto &function, bool codec = false);
void reject_external_struct_data(const TypeProto &type);

void reject_external_codec_tensor(const TensorProto &tensor) {
  if ((tensor.has_data_location() &&
       tensor.data_location() == TensorProto::DataLocation::EXTERNAL) ||
      !tensor.external_data().empty())
    throw std::runtime_error("External tensor data in struct declarations/codecs is unsupported "
                             "for encrypted storage: " +
                             std::string(tensor.name()));
}

void reject_external_codec_tensor(const SparseTensorProto &tensor) {
  if (tensor.has_values())
    reject_external_codec_tensor(tensor.values());
  if (tensor.has_indices())
    reject_external_codec_tensor(tensor.indices());
}

void reject_external_struct_data(const StructTypeProto &type) {
  if (type.has_decoder())
    reject_unsupported_external_data(type.decoder(), true);
  if (type.has_encoder())
    reject_unsupported_external_data(type.encoder(), true);
  if (type.has_array() && type.array().has_element_type())
    reject_external_struct_data(type.array().element_type());
  if (type.has_structure()) {
    for (const auto &field : type.structure().field()) {
      if (field.has_type())
        reject_external_struct_data(field.type());
      if (field.has_constant())
        reject_external_codec_tensor(field.constant());
    }
  }
}

void reject_external_struct_data(const TypeProto &type) {
  if (type.has_struct_type())
    reject_external_struct_data(type.struct_type());
  if (type.has_sequence_type() && type.sequence_type().has_elem_type())
    reject_external_struct_data(type.sequence_type().elem_type());
  if (type.has_optional_type() && type.optional_type().has_elem_type())
    reject_external_struct_data(type.optional_type().elem_type());
  if (type.has_map_type() && type.map_type().has_value_type())
    reject_external_struct_data(type.map_type().value_type());
}

void reject_external_struct_data(const EncodedValueProto &value) {
  if (value.has_struct_type())
    reject_external_struct_data(value.struct_type());
  if (value.has_logical_type())
    reject_external_struct_data(value.logical_type());
}

void reject_external_encoded_value(const EncodedValueProto &value, bool codec) {
  if ((value.has_data_location() && value.data_location() == TensorProto::DataLocation::EXTERNAL) ||
      !value.external_data().empty())
    throw std::runtime_error("External encoded initializer payloads are unsupported for "
                             "encrypted export: " +
                             std::string(value.name()));
  if (codec && value.has_affine()) {
    if (value.affine().has_scale())
      reject_external_codec_tensor(value.affine().scale());
    if (value.affine().has_zero_point())
      reject_external_codec_tensor(value.affine().zero_point());
  }
  reject_external_struct_data(value);
}

void reject_external_struct_data(const utils::RepeatedProtoField<ValueInfoProto> &values) {
  for (const auto &value : values)
    if (value.has_type())
      reject_external_struct_data(value.type());
}

void reject_unsupported_external_data(const AttributeProto &attribute, bool codec) {
  if (codec) {
    if (attribute.has_t())
      reject_external_codec_tensor(attribute.t());
    for (const auto &tensor : attribute.tensors())
      reject_external_codec_tensor(tensor);
    if (attribute.has_sparse_tensor())
      reject_external_codec_tensor(attribute.sparse_tensor());
    for (const auto &tensor : attribute.sparse_tensors())
      reject_external_codec_tensor(tensor);
  }
  if (attribute.has_tp())
    reject_external_struct_data(attribute.tp());
  for (const auto &type : attribute.type_protos())
    reject_external_struct_data(type);
  if (attribute.has_g())
    reject_unsupported_external_data(attribute.g(), codec);
  for (const auto &graph : attribute.graphs())
    reject_unsupported_external_data(graph, codec);
}

void reject_unsupported_external_data(const NodeProto &node, bool codec = false) {
  for (const auto &attribute : node.attribute())
    reject_unsupported_external_data(attribute, codec);
}

void reject_unsupported_external_data(const FunctionProto &function, bool codec) {
  reject_external_struct_data(function.value_info());
  for (const auto &node : function.node())
    reject_unsupported_external_data(node, codec);
  for (const auto &attribute : function.attribute_proto())
    reject_unsupported_external_data(attribute, codec);
}

void reject_unsupported_external_data(const GraphProto &graph, bool codec) {
  if (codec) {
    for (const auto &tensor : graph.initializer())
      reject_external_codec_tensor(tensor);
    for (const auto &tensor : graph.sparse_initializer())
      reject_external_codec_tensor(tensor);
  }
  reject_external_struct_data(graph.input());
  reject_external_struct_data(graph.output());
  reject_external_struct_data(graph.value_info());
  for (const auto &initializer : graph.extended_initializer()) {
    if (initializer.has_encoded_value())
      reject_external_encoded_value(initializer.encoded_value(), codec);
    if (initializer.has_paged_cache()) {
      for (const auto &block : initializer.paged_cache().blocks()) {
        if (codec) {
          if (block.has_key())
            reject_external_codec_tensor(block.key());
          if (block.has_value())
            reject_external_codec_tensor(block.value());
          if (block.has_encoded_key())
            reject_external_encoded_value(block.encoded_key(), true);
          if (block.has_encoded_value())
            reject_external_encoded_value(block.encoded_value(), true);
        } else {
          if (block.has_encoded_key())
            reject_external_struct_data(block.encoded_key());
          if (block.has_encoded_value())
            reject_external_struct_data(block.encoded_value());
        }
      }
    }
  }
  for (const auto &node : graph.node())
    reject_unsupported_external_data(node, codec);
}

void reject_unsupported_external_data(const ModelProto &model) {
  for (const auto &type : model.struct_types())
    reject_external_struct_data(type);
  if (model.has_graph())
    reject_unsupported_external_data(model.graph());
  for (const auto &function : model.functions())
    reject_unsupported_external_data(function);
  for (const auto &entry : model.rewritings()) {
    const RewritingProto &rewriting = entry.second;
    for (const auto &function : rewriting.added_functions())
      reject_unsupported_external_data(function);
    for (const auto &rewrite : rewriting.rewrites()) {
      for (const auto &node : rewrite.added_nodes())
        reject_unsupported_external_data(node);
      for (const auto &node : rewrite.initializer_calls())
        reject_unsupported_external_data(node);
    }
  }
}

} // namespace

void SaveEncryptedModelWithExternalData(ModelProto &model, const std::string &file_path,
                                        const std::string &key, size_t max_external_file_size,
                                        size_t size_threshold) {
  validate_external_limits(key, max_external_file_size);
  const size_t capacity = max_external_file_size - GCM_HEADER_SIZE;
  reject_unsupported_external_data(model);
  ModelProto copy = utils::CopyWithSharedTensorStorage(model);
  IteratorTensorProto check(&copy);
  while (check.next()) {
    TensorProto &tensor = *check;
    if (tensor.has_data_location() &&
        tensor.data_location() == TensorProto::DataLocation::EXTERNAL) {
      if (!tensor.has_raw_data())
        throw std::runtime_error("External tensor data must be loaded before encrypted export: " +
                                 std::string(tensor.name()));
      tensor.clr_external_data();
      tensor.set_data_location(TensorProto::DataLocation::DEFAULT);
    }
    if (tensor.raw_data().size() >= size_threshold && tensor.raw_data().size() > capacity)
      throw std::runtime_error("Tensor exceeds encrypted external file size limit: " +
                               std::string(tensor.name()));
  }

  uint8_t identifier[16];
  if (RAND_bytes(identifier, sizeof(identifier)) != 1)
    throw std::runtime_error("RAND_bytes failed: " + openssl_last_error());
  const std::filesystem::path path(file_path);
  const std::string prefix = utils::detail::external_path_to_utf8(path.filename()) + ".weights." +
                             hex_bytes(identifier, sizeof(identifier)) + ".";
  std::string plain;
  std::vector<TensorProto *> tensors;
  size_t index = 0;
  auto flush = [&]() {
    if (tensors.empty())
      return;
    const std::string location = prefix + std::to_string(index++) + ".onnxc";
    const std::string blob = encrypt_to_blob_gcm(plain, key);
    const std::string digest = encrypted_digest(blob);
    write_encrypted_file(path.parent_path() /
                             std::filesystem::path(std::u8string(location.begin(), location.end())),
                         blob);
    for (TensorProto *tensor : tensors) {
      add_external_entry(*tensor, kExternalDataLocationKey, location);
      add_external_entry(*tensor, kExternalDataEncryptionKey, kExternalDataEncryptionFormat);
      add_external_entry(*tensor, kExternalDataDigestKey, digest);
    }
    tensors.clear();
    plain.clear();
  };
  IteratorTensorProto iterator(&copy);
  while (iterator.next()) {
    TensorProto &tensor = *iterator;
    const size_t size = tensor.raw_data().size();
    if (size == 0 || size < size_threshold)
      continue;
    if (size > capacity - plain.size())
      flush();
    tensor.clr_external_data();
    add_external_entry(tensor, kExternalDataOffsetKey, std::to_string(plain.size()));
    add_external_entry(tensor, kExternalDataLengthKey, std::to_string(size));
    const auto &raw_data = tensor.raw_data();
    plain.append(reinterpret_cast<const char *>(raw_data.data()), size);
    tensor.clear_raw_data();
    tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
    tensors.push_back(&tensor);
  }
  flush();
  write_encrypted_file(path, SaveEncryptedModelToString(copy, key, "AES-256-GCM"));
}

void LoadEncryptedExternalDataForModel(ModelProto &model, const std::string &base_dir,
                                       const std::string &key, const ParseOptions &opts,
                                       size_t max_external_file_size) {
  validate_external_limits(key, max_external_file_size);
  onnx_light_helpers::ValidateAlignmentOption(opts.alignment, "ParseOptions.alignment");
  struct TensorReference {
    TensorProto *tensor;
    GraphProto *graph;
    uint64_t offset;
    uint64_t length;
  };
  struct Shard {
    std::string digest;
    std::vector<TensorReference> tensors;
  };
  std::map<std::string, Shard> shards;
  IteratorTensorProto iterator(&model);
  while (iterator.next()) {
    TensorProto &tensor = *iterator;
    bool marked = false;
    std::string location;
    for (const auto &entry : tensor.external_data()) {
      marked |= entry.key() == kExternalDataEncryptionKey;
      if (entry.key() == kExternalDataLocationKey && location.empty())
        location = entry.value();
    }
    if (!marked)
      continue;
    std::map<std::string, std::string> entries;
    for (const auto &entry : tensor.external_data()) {
      if (!entries.emplace(std::string(entry.key()), std::string(entry.value())).second)
        throw std::runtime_error("Duplicate encrypted external-data key '" +
                                 std::string(entry.key()) + "' for " +
                                 external_tensor_context(tensor, location));
    }
    const auto invalid = [&](const std::string &field) {
      return std::runtime_error("Invalid encrypted external-data " + field + " for " +
                                external_tensor_context(tensor, location));
    };
    const auto required = [&](const char *field) -> const std::string & {
      const auto found = entries.find(field);
      if (found == entries.end())
        throw invalid(std::string("missing field '") + field + "'");
      return found->second;
    };
    int64_t offset = -1, length = -1;
    if (tensor.data_location() != TensorProto::DataLocation::EXTERNAL)
      throw invalid("data_location");
    if (required(kExternalDataEncryptionKey) != kExternalDataEncryptionFormat)
      throw invalid(kExternalDataEncryptionKey);
    if (!IsSafeExternalDataLocation(required(kExternalDataLocationKey)))
      throw invalid(kExternalDataLocationKey);
    const std::string &digest = required(kExternalDataDigestKey);
    if (digest.size() != 64)
      throw invalid(kExternalDataDigestKey);
    if (!parse_external_integer(required(kExternalDataOffsetKey), offset) || offset < 0)
      throw invalid(kExternalDataOffsetKey);
    if (!parse_external_integer(required(kExternalDataLengthKey), length) || length < 0)
      throw invalid(kExternalDataLengthKey);
    if (opts.max_tensor_size_bytes > 0 && length > opts.max_tensor_size_bytes)
      throw invalid("length exceeding max_tensor_size_bytes");
    auto &shard = shards[location];
    if (shard.tensors.empty())
      shard.digest = digest;
    else if (shard.digest != digest)
      throw invalid(std::string("inconsistent ") + kExternalDataDigestKey);
    shard.tensors.push_back(TensorReference{
        &tensor, iterator.graph(), static_cast<uint64_t>(offset), static_cast<uint64_t>(length)});
  }
  for (const auto &[location, shard] : shards) {
    std::vector<uint8_t> plain;
    {
      auto file = utils::detail::OpenExternalDataFile(base_dir, location);
      const int64_t size = utils::detail::ExternalDataFileSize(*file);
      if (size < static_cast<int64_t>(GCM_HEADER_SIZE) ||
          static_cast<uint64_t>(size) > max_external_file_size)
        throw std::runtime_error("Encrypted external file exceeds size limit or is truncated for " +
                                 external_tensor_context(*shard.tensors.front().tensor, location));
      std::string blob(static_cast<size_t>(size), '\0');
      utils::detail::ReadExternalDataFile(*file, blob.data(), size, 0);
      const std::string digest = encrypted_digest(blob);
      if (digest != shard.digest ||
          blob.compare(0, MAGIC_GCM.size(), MAGIC_GCM.data(), MAGIC_GCM.size()) != 0)
        throw std::runtime_error(
            "Encrypted external file authentication failed (digest or format) for " +
            external_tensor_context(*shard.tensors.front().tensor, location));
      plain =
          decrypt_from_blob_gcm(reinterpret_cast<const uint8_t *>(blob.data()), blob.size(), key);
    }
    for (const auto &reference : shard.tensors) {
      if (reference.offset > plain.size() || reference.length > plain.size() - reference.offset)
        throw std::runtime_error("Encrypted external tensor offset/length range is invalid for " +
                                 external_tensor_context(*reference.tensor, location));
      const size_t offset = static_cast<size_t>(reference.offset);
      const size_t length = static_cast<size_t>(reference.length);
      TensorProto &tensor = *reference.tensor;
      if (opts.alignment > 1)
        tensor.ref_raw_data().resize_aligned(length, static_cast<size_t>(opts.alignment));
      else
        tensor.ref_raw_data().resize(length);
      if (length > 0)
        std::memcpy(tensor.ref_raw_data().data(), plain.data() + offset, length);
      tensor.clr_external_data();
      tensor.set_data_location(TensorProto::DataLocation::DEFAULT);
      if (opts.raw_data_callback && tensor.has_raw_data()) {
        auto deleter = opts.raw_data_callback(tensor, reference.graph);
        if (deleter)
          tensor.ref_raw_data().attach_deleter(std::move(deleter));
      }
    }
  }
}

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
  bool encrypted_external = false;
  IteratorTensorProto iterator(&model);
  while (!encrypted_external && iterator.next()) {
    for (const auto &entry : iterator->external_data()) {
      if (entry.key() == kExternalDataEncryptionKey) {
        encrypted_external = true;
        break;
      }
    }
  }
  if (encrypted_external) {
    if (file_buf.compare(0, MAGIC_GCM.size(), MAGIC_GCM.data(), MAGIC_GCM.size()) != 0)
      throw std::runtime_error("Encrypted external weights require an ONNXCRY3 model.");

    std::filesystem::path base_dir_path = std::filesystem::path(file_path).parent_path();
    if (base_dir_path.empty()) {
      base_dir_path = ".";
    }

    LoadEncryptedExternalDataForModel(model, utils::detail::external_path_to_utf8(base_dir_path), key,
                                     opts);
  }
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
  // The decrypted buffer is local, so parsed tensors must own their bytes.
  mutable_opts.no_copy = false;
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
