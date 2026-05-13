#include "onnx_crypt.h"

#ifdef ONNX_LIGHT_HAS_OPENSSL

#include "stream.h"
#include <array>
#include <cstring>
#include <fstream>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

// File format constants.
static constexpr std::array<char, 8> MAGIC{'O', 'N', 'N', 'X', 'C', 'R', 'Y', '1'};
static constexpr int SALT_LEN = 16;
static constexpr int IV_LEN = 16;
static constexpr int KEY_LEN = 32; // AES-256
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

} // namespace

void SaveEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const SerializeOptions &opts) {
  // 1. Serialize the model to memory.
  std::string plain;
  SerializeOptions mutable_opts = opts;
  model.SerializeToString(plain, mutable_opts);

  // 2. Generate random salt and IV.
  uint8_t salt[SALT_LEN];
  uint8_t iv[IV_LEN];
  if (RAND_bytes(salt, SALT_LEN) != 1 || RAND_bytes(iv, IV_LEN) != 1) {
    throw std::runtime_error("RAND_bytes failed: " + openssl_last_error());
  }

  // 3. Derive AES-256 key.
  auto aes_key = derive_key(key, salt);

  // 4. Encrypt using AES-256-CBC (PKCS#7 padding).
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

  // 5. Write header + ciphertext to file.
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open())
    throw std::runtime_error("Cannot open file for writing: " + file_path);

  ofs.write(MAGIC.data(), static_cast<std::streamsize>(MAGIC.size()));
  ofs.write(reinterpret_cast<const char *>(salt), SALT_LEN);
  ofs.write(reinterpret_cast<const char *>(iv), IV_LEN);
  ofs.write(reinterpret_cast<const char *>(ciphertext.data()),
            static_cast<std::streamsize>(cipher_len));

  if (!ofs)
    throw std::runtime_error("Write error for file: " + file_path);
}

void LoadEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const ParseOptions &opts) {
  // 1. Read the entire file.
  std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open encrypted file: " + file_path);

  const std::streamsize file_size = ifs.tellg();
  const std::streamsize header_size =
      static_cast<std::streamsize>(MAGIC.size() + SALT_LEN + IV_LEN);
  if (file_size < header_size)
    throw std::runtime_error("File too small to be a valid encrypted model: " + file_path);

  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> file_buf(static_cast<size_t>(file_size));
  ifs.read(reinterpret_cast<char *>(file_buf.data()), file_size);
  if (!ifs)
    throw std::runtime_error("Read error for file: " + file_path);

  // 2. Check magic.
  if (std::memcmp(file_buf.data(), MAGIC.data(), MAGIC.size()) != 0)
    throw std::runtime_error("Bad magic bytes – not an ONNXCRY1 encrypted file: " + file_path);

  const uint8_t *salt = file_buf.data() + MAGIC.size();
  const uint8_t *iv = salt + SALT_LEN;
  const uint8_t *ciphertext = iv + IV_LEN;
  const size_t cipher_len = static_cast<size_t>(file_size) - MAGIC.size() - SALT_LEN - IV_LEN;

  // 3. Derive AES-256 key.
  uint8_t salt_arr[SALT_LEN];
  std::memcpy(salt_arr, salt, SALT_LEN);
  auto aes_key = derive_key(key, salt_arr);

  // 4. Decrypt.
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx)
    throw std::runtime_error("EVP_CIPHER_CTX_new failed: " + openssl_last_error());

  if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, aes_key.data(), iv) != 1) {
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
    throw std::runtime_error("EVP_DecryptFinal_ex failed (wrong key or corrupt file): " +
                             openssl_last_error());
  }
  EVP_CIPHER_CTX_free(ctx);
  const size_t plain_len = static_cast<size_t>(out_len + final_len);

  // 5. Parse the decrypted protobuf.
  ParseOptions mutable_opts = opts;
  utils::StringStream stream(plaintext.data(), static_cast<int64_t>(plain_len));
  model.ParseFromStream(stream, mutable_opts);
}

} // namespace ONNX_LIGHT_NAMESPACE

#endif // ONNX_LIGHT_HAS_OPENSSL
