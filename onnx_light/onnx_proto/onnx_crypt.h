#pragma once

#include "onnx.h"
#include "onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {

#ifdef ONNX_LIGHT_HAS_OPENSSL

/**
 * @defgroup encryption Encrypted model I/O
 * @{
 *
 * Saves and loads ONNX ModelProto objects as single, AES-256-CBC encrypted
 * binary files.
 *
 * @warning **Security limitation**: The current format (ONNXCRY1) uses
 * AES-256-CBC without a Message Authentication Code (MAC).  This means
 * encrypted payloads are malleable — an attacker who can modify the
 * ciphertext can tamper with the decrypted output without detection.
 * A future format revision (ONNXCRY2) should use an authenticated
 * encryption mode (e.g. AES-256-GCM) to provide integrity guarantees.
 * Until then, callers should verify model integrity via an external
 * mechanism (e.g. HMAC signature, content hash) when loading models
 * from untrusted sources.
 *
 * ### File format
 *
 * ```
 * Offset  Size  Field
 * ------  ----  -----
 *      0     8  Magic: "ONNXCRY1"
 *      8    16  Random PBKDF2 salt
 *     24    16  Random AES-CBC initialisation vector
 *     40     N  AES-256-CBC ciphertext (PKCS#7 padded protobuf payload)
 * ```
 *
 * ### Key derivation
 * The caller-supplied passphrase is stretched to a 32-byte AES key using
 * PBKDF2-HMAC-SHA256 (100 000 iterations, RFC 2898).
 * Passing a raw 32-byte binary key is also valid; PBKDF2 is always applied
 * so the on-disk representation is fully self-contained.
 */

/**
 * Serializes *model* to a single AES-256-CBC encrypted file.
 *
 * The ModelProto is first serialized to an in-memory buffer (honouring
 * *opts*), then encrypted and written to *file_path*.  The input model is
 * left unchanged.
 *
 * @param model   The model to save.
 * @param file_path  Destination file path.  Created or truncated.
 * @param key     Passphrase / raw key used to derive the AES-256 key via
 *                PBKDF2-HMAC-SHA256 (100 000 iterations).
 * @param opts    Serialization options (e.g. raw_data_threshold).
 * @throws std::runtime_error on OpenSSL errors or I/O failures.
 */
void SaveEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const SerializeOptions &opts = SerializeOptions{});

/**
 * Loads and decrypts an AES-256-CBC encrypted ONNX model from *file_path*.
 *
 * The file must have been produced by SaveEncryptedModel() with the same
 * passphrase.  The decrypted bytes are parsed into *model*.
 *
 * @param model      Output model populated from the decrypted payload.
 * @param file_path  Source file path.
 * @param key        Passphrase / raw key (must match the one used to save).
 * @param opts       Parsing options.
 * @throws std::runtime_error on decryption failure, bad magic, or I/O errors.
 */
void LoadEncryptedModel(ModelProto &model, const std::string &file_path, const std::string &key,
                        const ParseOptions &opts = ParseOptions{});

/**
 * Serializes *model* to an in-memory AES-256-CBC encrypted byte string.
 *
 * Equivalent to SaveEncryptedModel() but the ciphertext is returned as a
 * `std::string` (raw bytes) instead of being written to a file.  The caller
 * can write, transmit, or cache the returned buffer as required.
 *
 * @param model  The model to encrypt.
 * @param key    Passphrase / raw key used to derive the AES-256 key via
 *               PBKDF2-HMAC-SHA256 (100 000 iterations).
 * @param opts   Serialization options.
 * @return       Raw encrypted bytes in the ONNXCRY1 format.
 * @throws std::runtime_error on OpenSSL errors.
 */
std::string SaveEncryptedModelToString(ModelProto &model, const std::string &key,
                                       const SerializeOptions &opts = SerializeOptions{});

/**
 * Decrypts and parses an in-memory AES-256-CBC encrypted byte string into
 * *model*.
 *
 * The buffer must have been produced by SaveEncryptedModelToString() (or
 * SaveEncryptedModel()) with the same passphrase.
 *
 * @param model          Output model populated from the decrypted payload.
 * @param encrypted_data Raw encrypted bytes in the ONNXCRY1 format.
 * @param key            Passphrase / raw key (must match the one used to save).
 * @param opts           Parsing options.
 * @throws std::runtime_error on decryption failure or bad magic.
 */
void LoadEncryptedModelFromString(ModelProto &model, const std::string &encrypted_data,
                                  const std::string &key,
                                  const ParseOptions &opts = ParseOptions{});

/** @} */

#endif // ONNX_LIGHT_HAS_OPENSSL

} // namespace ONNX_LIGHT_NAMESPACE
