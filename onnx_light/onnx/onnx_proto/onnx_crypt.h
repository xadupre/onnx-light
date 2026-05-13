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

/** @} */

#endif // ONNX_LIGHT_HAS_OPENSSL

} // namespace ONNX_LIGHT_NAMESPACE
