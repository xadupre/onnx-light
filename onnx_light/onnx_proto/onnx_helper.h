#pragma once

#include "onnx.h"

#include <initializer_list>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * The function populates external data for every tensor.
 * The function does not remove anything from the model.
 * @param model Model to update.
 * @param threshold Minimum raw_data size (in bytes) to switch to external storage.
 * @param external_data_location Relative or absolute path to the external weights file.
 * @param use_external_data_location If true, tensors already marked as EXTERNAL keep their current
 *        external_data.location instead of being reassigned.
 * @param max_external_file_size Maximum size in bytes for one external weights file.
 *        If > 0, tensors are split across multiple files by appending ``.1``, ``.2``...
 * @param alignment If > 0, each tensor's offset within its weights file is rounded up to
 *        the nearest multiple of alignment bytes.  Use 4096 for mmap-friendly page alignment.
 * @return The total number of bytes in the external weights file(s), including any padding.
 */
offset_t PopulateExternalData(ModelProto &model, size_t threshold,
                              const std::string &external_data_location,
                              bool use_external_data_location = true,
                              int64_t max_external_file_size = 0, int64_t alignment = 0);

/**
 * Clears the external data from the model.
 * @param model Model to update.
 */
void ClearExternalData(ModelProto &model);

/**
 * Transfers all tensor raw_data whose size is >= opts.raw_data_threshold into a single
 * contiguous buffer owned via a shared_ptr, updating each qualifying tensor's raw_data
 * to borrow from that buffer.  The buffer is kept alive by the shared_ptr stored inside
 * each tensor's ByteSpan; the caller does not need to retain the returned shared_ptr
 * for the tensors to remain valid.
 *
 * Mirrors the no-copy external-data loading scenario: each tensor borrows a slice
 * of a single shared buffer, avoiding per-tensor allocations.
 *
 * @param model Model whose tensors will be consolidated in-place.
 * @param opts  Options controlling the size threshold and byte alignment.
 *              - raw_data_threshold: only tensors with raw_data.size() >= this value are moved.
 *              - alignment: if > 0, each tensor's offset is padded to a multiple of this value.
 * @return      Shared ownership handle for the consolidated buffer, or nullptr if no tensors
 *              qualified.  The buffer lifetime is also managed by the individual tensors.
 */
std::shared_ptr<uint8_t[]>
ConsolidateTensorsToBuffer(ModelProto &model,
                           const TensorBufferOptions &opts = TensorBufferOptions{});

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
 * When external weights are written, temporary external_data metadata is
 * removed by default (clear_external_data=true), so two-file serialization
 * leaves ModelProto unchanged after the call.
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
 * When external weights are written, temporary external_data metadata is
 * removed by default (clear_external_data=true), so two-file serialization
 * leaves ModelProto unchanged after the call.
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

////////////////////
// Input/output helpers
////////////////////

/**
 * Appends a batch of input names to ``proto`` in a single call.
 *
 * Works with any ONNX proto exposing an ``add_input`` member that accepts the
 * elements of ``names`` (typically ``NodeProto``, ``FunctionProto`` and
 * any other proto with a ``FIELD_REPEATED_STR(input, ...)`` field). Allows
 * passing an ``std::initializer_list<const char *>``, ``std::vector<std::string>``
 * or any other range whose elements are accepted by ``add_input``.
 *
 * @tparam ProtoT  ONNX proto type with an ``add_input`` member function.
 * @tparam Range   Range whose elements are accepted by ``ProtoT::add_input``.
 * @param  proto   Proto to append names to.
 * @param  names   Range of input names to append, in order.
 */
template <typename ProtoT, typename Range>
inline void AddInputs(ProtoT &proto, const Range &names) {
  for (const auto &name : names) {
    proto.add_input(name);
  }
}

/// initializer_list overload of :ref:`AddInputs` so call sites can pass a
/// brace-enclosed list of names directly (e.g. ``AddInputs(node, {"a", "b"})``)
/// without specifying the template arguments explicitly.
template <typename ProtoT, typename T>
inline void AddInputs(ProtoT &proto, std::initializer_list<T> names) {
  for (const auto &name : names) {
    proto.add_input(name);
  }
}

/**
 * Appends a batch of output names to ``proto`` in a single call. See
 * :ref:`AddInputs` for the requirements on ``ProtoT`` and ``Range``.
 *
 * @tparam ProtoT  ONNX proto type with an ``add_output`` member function.
 * @tparam Range   Range whose elements are accepted by ``ProtoT::add_output``.
 * @param  proto   Proto to append names to.
 * @param  names   Range of output names to append, in order.
 */
template <typename ProtoT, typename Range>
inline void AddOutputs(ProtoT &proto, const Range &names) {
  for (const auto &name : names) {
    proto.add_output(name);
  }
}

/// initializer_list overload of :ref:`AddOutputs`.
template <typename ProtoT, typename T>
inline void AddOutputs(ProtoT &proto, std::initializer_list<T> names) {
  for (const auto &name : names) {
    proto.add_output(name);
  }
}

/**
 * Appends a single FLOAT attribute (``name`` -> ``value``) to ``proto``.
 *
 * Works with any ONNX proto exposing an ``add_attribute`` member that returns
 * an ``AttributeProto *`` (typically ``NodeProto``).
 *
 * @tparam ProtoT  ONNX proto type with an ``add_attribute`` member function.
 * @param  proto   Proto to append the attribute to.
 * @param  name    Attribute name.
 * @param  value   Attribute float value.
 */
template <typename ProtoT>
inline void AddFloatAttribute(ProtoT &proto, const char *name, float value) {
  AttributeProto *attr = proto.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(value);
}

} // namespace ONNX_LIGHT_NAMESPACE
