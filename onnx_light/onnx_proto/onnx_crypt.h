#pragma once

#include "onnx.h"
#include "onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {

#ifdef ONNX_LIGHT_HAS_OPENSSL

/**
 * @defgroup encryption Encrypted model I/O
 * @{
 *
 * Saves and loads ONNX ModelProto objects as single encrypted binary files.
 *
 * Supported formats:
 * - ONNXCRY1: AES-256-CBC (legacy, no MAC)
 * - ONNXCRY2: ChaCha20-Poly1305 (authenticated encryption)
 *
 * ### File format
 *
 * ```
 * Offset  Size  Field
 * ------  ----  -----
 *   ONNXCRY1 (AES-256-CBC):
 *      0     8  Magic: "ONNXCRY1"
 *      8    16  Random PBKDF2 salt
 *     24    16  Random AES-CBC initialisation vector
 *     40     N  AES-256-CBC ciphertext (PKCS#7 padded protobuf payload)
 *
 *   ONNXCRY2 (ChaCha20-Poly1305):
 *      0     8  Magic: "ONNXCRY2"
 *      8    16  Random PBKDF2 salt
 *     24    12  Random nonce
 *     36    16  Authentication tag
 *     52     N  ChaCha20 ciphertext (same length as plaintext)
 * ```
 *
 * ### Key derivation
 * The caller-supplied passphrase is stretched to a 32-byte AES key using
 * PBKDF2-HMAC-SHA256 (100 000 iterations, RFC 2898).
 * Passing a raw 32-byte binary key is also valid; PBKDF2 is always applied
 * so the on-disk representation is fully self-contained.
 */

/**
 * Serializes *model* to a single encrypted file.
 *
 * The ModelProto is first serialized to an in-memory buffer (honouring
 * *opts*), then encrypted and written to *file_path*.  The input model is
 * left unchanged.
 *
 * @param model   The model to save.
 * @param file_path  Destination file path.  Created or truncated.
 * @param key     Passphrase / raw key used to derive the encryption key via
 *                PBKDF2-HMAC-SHA256 (100 000 iterations).
 * @param opts    Serialization options (e.g. raw_data_threshold).
 * @param encryption  Encryption algorithm: ``"AES-256-CBC"`` (ONNXCRY1) or
 *                ``"ChaCha20-Poly1305"`` (ONNXCRY2).
 * @throws std::runtime_error on OpenSSL errors or I/O failures.
 */
ONNX_LIGHT_PROTO_API void SaveEncryptedModel(ModelProto &model, const std::string &file_path,
                                             const std::string &key,
                                             const SerializeOptions &opts = SerializeOptions{},
                                             const std::string &encryption = "AES-256-CBC");
ONNX_LIGHT_PROTO_API void SaveEncryptedModel(ModelProto &model, const std::string &file_path,
                                             const std::string &key, const std::string &encryption);

/**
 * Loads and decrypts an encrypted ONNX model from *file_path*.
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
ONNX_LIGHT_PROTO_API void LoadEncryptedModel(ModelProto &model, const std::string &file_path,
                                             const std::string &key,
                                             const ParseOptions &opts = ParseOptions{});

/**
 * Serializes *model* to an in-memory encrypted byte string.
 *
 * Equivalent to SaveEncryptedModel() but the ciphertext is returned as a
 * `std::string` (raw bytes) instead of being written to a file.  The caller
 * can write, transmit, or cache the returned buffer as required.
 *
 * @param model  The model to encrypt.
 * @param key    Passphrase / raw key used to derive the encryption key via
 *               PBKDF2-HMAC-SHA256 (100 000 iterations).
 * @param opts   Serialization options.
 * @param encryption  Encryption algorithm: ``"AES-256-CBC"`` (ONNXCRY1) or
 *               ``"ChaCha20-Poly1305"`` (ONNXCRY2).
 * @return       Raw encrypted bytes in ONNXCRY1 or ONNXCRY2 format.
 * @throws std::runtime_error on OpenSSL errors.
 */
ONNX_LIGHT_PROTO_API std::string
SaveEncryptedModelToString(ModelProto &model, const std::string &key,
                           const SerializeOptions &opts = SerializeOptions{},
                           const std::string &encryption = "AES-256-CBC");
ONNX_LIGHT_PROTO_API std::string SaveEncryptedModelToString(ModelProto &model,
                                                            const std::string &key,
                                                            const std::string &encryption);

/**
 * Decrypts and parses an in-memory encrypted byte string into
 * *model*.
 *
 * The buffer must have been produced by SaveEncryptedModelToString() (or
 * SaveEncryptedModel()) with the same passphrase.
 *
 * @param model          Output model populated from the decrypted payload.
 * @param encrypted_data Raw encrypted bytes in ONNXCRY1 or ONNXCRY2 format.
 * @param key            Passphrase / raw key (must match the one used to save).
 * @param opts           Parsing options.
 * @throws std::runtime_error on decryption failure or bad magic.
 */
ONNX_LIGHT_PROTO_API void LoadEncryptedModelFromString(ModelProto &model,
                                                       const std::string &encrypted_data,
                                                       const std::string &key,
                                                       const ParseOptions &opts = ParseOptions{});

/** @} */

#endif // ONNX_LIGHT_HAS_OPENSSL

} // namespace ONNX_LIGHT_NAMESPACE
