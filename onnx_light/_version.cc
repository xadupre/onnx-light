/**
 * @file _version.cc
 * @brief Returns the version string of the onnx-light package.
 */

#include <string>

namespace onnx_light {

/**
 * Returns the package version string.
 *
 * @return Version string "0.1.0".
 */
const std::string &version() {
  static const std::string kVersion = "0.1.0";
  return kVersion;
}

} // namespace onnx_light
