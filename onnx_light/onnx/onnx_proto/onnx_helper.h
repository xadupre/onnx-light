#pragma once

#include "onnx.h"

namespace onnx {

/**
 * The function populates external data for every tensor.
 * The function does not remove anything from the model.
 * @param model Model to update.
 * @param threshold Minimum raw_data size (in bytes) to switch to external storage.
 * @param external_data_location Relative or absolute path to the external weights file.
 * @return The total number of bytes written to the external data file.
 */
offset_t PopulateExternalData(ModelProto &model, size_t threshold,
                              const std::string &external_data_location);

/**
 * Clears the external data from the model.
 * @param model Model to update.
 */
void ClearExternalData(ModelProto &model);

/**
 * IteratorTensorProto is an iterator that traverses all TensorProto objects.
 */
class IteratorTensorProto {
protected:
  /**
   * Tracks traversal indices for one graph level in the DFS stack.
   */
  struct Position {
    /**
     * Points to the graph traversed at this stack level.
     */
    GraphProto *graph;
    /**
     * Stores the current node index in graph->ref_node().
     */
    int node_index = 0;
    /**
     * Stores the current attribute index in node->ref_attribute().
     */
    int attr_index = 0;
    /**
     * Stores the current initializer index in graph->ref_initializer().
     */
    int node_initializer_index = 0;
  };

public:
  /**
   * Initializes the iterator from a graph root.
   * @param graph Root graph to traverse.
   */
  explicit inline IteratorTensorProto(GraphProto *graph) : tp_(nullptr), positions_() {
    positions_.emplace_back(Position{graph});
  }
  /**
   * Returns the current tensor reference.
   */
  inline TensorProto &operator*() { return *tp_; }
  /**
   * Returns the current tensor pointer.
   */
  inline TensorProto *operator->() { return tp_; }
  /**
   * Advances to the next tensor.
   * Returns true when one is found.
   */
  bool next();

private:
  /**
   * Stores the current tensor found by the traversal.
   */
  TensorProto *tp_;
  /**
   * Stores the DFS traversal stack.
   */
  std::vector<Position> positions_;
};

//////////////
// Serializing
//////////////

/**
 * The function saves the ONNX model to a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 * @tparam T ONNX proto type to serialize.
 * @param stream Output stream.
 * @param options Serialization options.
 * @param clear_external_data If true, removes temporary external_data metadata after
 * serialization.
 */
template <typename T>
inline void SerializeProtoToStream(T &, utils::BinaryWriteStream &, SerializeOptions &,
                                   bool clear_external_data = true) {
  EXT_THROW("SerializeProtoToStream is not implemented for type ", typeid(T).name(),
            ", clear_external_data=", clear_external_data);
}

/**
 * The function saves the ONNX model to a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 * @param model Model to serialize.
 * @param stream Output stream.
 * @param options Serialization options.
 * @param clear_external_data If true, removes temporary external_data metadata after
 * serialization.
 */
void SerializeModelProtoToStream(ModelProto &model, utils::BinaryWriteStream &stream,
                                 SerializeOptions &options, bool clear_external_data = true);

/**
 * Specializes SerializeProtoToStream for ModelProto.
 */
template <>
inline void SerializeProtoToStream(ModelProto &model, utils::BinaryWriteStream &stream,
                                   SerializeOptions &options, bool clear_external_data) {
  SerializeModelProtoToStream(model, stream, options, clear_external_data);
}

//////////
// Parsing
//////////

/**
 * The function reads the ONNX model from a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 * @tparam T ONNX proto type to parse.
 * @param stream Input stream.
 * @param options Parsing options.
 * @param clear_external_data If true, removes temporary external_data metadata after parsing.
 */
template <typename T>
inline void ParseProtoFromStream(T &, utils::BinaryStream &, ParseOptions &,
                                 bool clear_external_data = true) {
  EXT_THROW("ParseProtoFromStream is not implemented for type ", typeid(T).name(),
            ", clear_external_data=", clear_external_data);
}

/**
 * The function reads the ONNX model from a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 * @param model Model to parse.
 * @param stream Input stream.
 * @param options Parsing options.
 * @param clear_external_data If true, removes temporary external_data metadata after parsing.
 */
void ParseModelProtoFromStream(ModelProto &model, utils::BinaryStream &stream,
                               ParseOptions &options, bool clear_external_data = true);

/**
 * Specializes ParseProtoFromStream for ModelProto.
 */
template <>
inline void ParseProtoFromStream(ModelProto &model, utils::BinaryStream &stream,
                                 ParseOptions &options, bool clear_external_data) {
  ParseModelProtoFromStream(model, stream, options, clear_external_data);
}

} // namespace onnx
