#pragma once

#include "onnx.h"
#include "onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {

#ifdef ONNX_LIGHT_HAS_OPENSSL

/**
 * @defgroup encryption Encrypted model I/O
 * @{
 *
 * Saves and loads ONNX ModelProto objects and optional encrypted external weights.
 *
 * Supported formats:
 * - ONNXCRY1: AES-256-CBC (legacy, no MAC)
 * - ONNXCRY2: ChaCha20-Poly1305 (authenticated encryption)
 * - ONNXCRY3: AES-256-GCM (authenticated encryption; header, salt and nonce are AAD)
 * Use ONNXCRY3 or ONNXCRY2 for new files; ONNXCRY1 does not authenticate the payload.
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
 *
 *   ONNXCRY3 (AES-256-GCM):
 *      0     8  Magic: "ONNXCRY3"
 *      8    16  Random PBKDF2 salt
 *     24    12  Random nonce
 *     36    16  Authentication tag
 *     52     N  AES-256-GCM ciphertext (same length as plaintext)
 *   The first 36 bytes are authenticated as additional data.
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
 * @param encryption  Encryption algorithm: ``"AES-256-CBC"`` (ONNXCRY1),
 *                ``"ChaCha20-Poly1305"`` (ONNXCRY2), or ``"AES-256-GCM"`` (ONNXCRY3).
 * @throws std::runtime_error on OpenSSL errors or I/O failures.
 */
ONNX_LIGHT_PROTO_API void SaveEncryptedModel(ModelProto &model, const std::string &file_path,
                                             const std::string &key,
                                             const SerializeOptions &opts = SerializeOptions{},
                                             const std::string &encryption = "AES-256-CBC");
ONNX_LIGHT_PROTO_API void SaveEncryptedModel(ModelProto &model, const std::string &file_path,
                                             const std::string &key, const std::string &encryption);

/**
 * Saves an ONNXCRY3 model with separately authenticated ONNXCRY3 weight files.
 *
 * Packs nonempty raw_data tensors of at least size_threshold bytes into shards
 * beside file_path. Typed tensor fields remain embedded. The input is unchanged;
 * existing external tensors must have their plaintext raw_data loaded first.
 * Struct declaration/codec tensors stay embedded; external references in those
 * scopes and external encoded initializers are rejected before writing files.
 * Metadata is copied, while safe tensor storage is shared with copy-on-write;
 * ownerless borrowed buffers and actively exported buffers are copied.
 * The caller must synchronize source access and reacquire existing storage views afterward.
 * No plaintext files are written. Shards have unique generated relative names.
 * max_external_file_size includes the 52-byte ONNXCRY3 envelope and must be
 * between 53 and INT_MAX + 52 bytes. Tensors are never split: an eligible tensor
 * exceeding the payload capacity throws before any file is written.
 *
 * Each tensor records location, offset and length in the decrypted shard, plus
 * onnx_light.encryption=ONNXCRY3 and onnx_light.encrypted_sha256 (the hex SHA-256
 * of the complete encrypted shard). The encrypted model authenticates this
 * metadata, preventing substitution with another valid shard under the same key.
 * Writes shards first and the model last; I/O failures may leave encrypted orphan
 * shards. Saving is not transactional; callers manage obsolete shards.
 *
 * @param model Model containing in-memory tensor data.
 * @param file_path Destination encrypted model path (created or truncated).
 * @param key Nonempty passphrase / binary key, shared by all files.
 * @param max_external_file_size Maximum encrypted shard size in bytes.
 * @param size_threshold Minimum raw_data size to externalize.
 * @throws std::runtime_error on invalid limits, unloaded external data, or I/O/crypto errors.
 */
ONNX_LIGHT_PROTO_API void SaveEncryptedModelWithExternalData(
    ModelProto &model, const std::string &file_path, const std::string &key,
    size_t max_external_file_size = 1024ULL * 1024 * 1024, size_t size_threshold = 1024);

/**
 * Loads authenticated external weights from an already decrypted model.
 *
 * Resolves marked external tensors relative to base_dir, verifies their
 * ciphertext SHA-256 against the authenticated model metadata, and decrypts each
 * selected shard once. Payload selection follows skip_raw_data and raw_data_threshold,
 * as for embedded tensors. Skipped tensors retain their external metadata; their
 * metadata, paths, file sizes and slice bounds are still validated, but shards with
 * no selected payloads are not read, authenticated or decrypted.
 * Offsets and lengths address plaintext, never ciphertext. Successfully loaded
 * tensors become embedded raw_data with external metadata removed. Unmarked
 * plain external tensors are left unchanged (LoadExternalDataForModel handles
 * those). The model metadata must come from a trusted/authenticated source.
 *
 * @param model Decrypted model whose marked tensors are populated in place.
 * @param base_dir Directory containing the encrypted model and its shards.
 * @param key Nonempty passphrase / binary key.
 * @param opts Parsing limits; max_tensor_size_bytes also applies to external slices.
 * @param max_external_file_size Maximum encrypted shard size accepted in bytes.
 * @throws std::runtime_error on wrong passwords, missing/corrupt/substituted files,
 *         unsafe paths, invalid metadata or limits. On failure, model may be partially loaded.
 */
ONNX_LIGHT_PROTO_API void
LoadEncryptedExternalDataForModel(ModelProto &model, const std::string &base_dir,
                                  const std::string &key, const ParseOptions &opts = ParseOptions{},
                                  size_t max_external_file_size = 1024ULL * 1024 * 1024);

/**
 * Loads and decrypts an encrypted ONNX model from *file_path*.
 *
 * The file must have been produced by SaveEncryptedModel() with the same
 * passphrase. The decrypted bytes are parsed into *model*. Marked encrypted
 * external weights are authenticated and embedded automatically, with a default
 * 1 GiB encrypted-shard limit. For a different limit, uses
 * LoadEncryptedModelFromString() followed by LoadEncryptedExternalDataForModel().
 * Plain external references retain their existing behavior (not loaded here).
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
 * @param encryption  Encryption algorithm: ``"AES-256-CBC"`` (ONNXCRY1),
 *               ``"ChaCha20-Poly1305"`` (ONNXCRY2), or ``"AES-256-GCM"`` (ONNXCRY3).
 * @return       Raw encrypted bytes in ONNXCRY1, ONNXCRY2, or ONNXCRY3 format.
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
 * @param encrypted_data Raw encrypted bytes in ONNXCRY1, ONNXCRY2, or ONNXCRY3 format.
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
