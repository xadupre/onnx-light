#pragma once

#include "onnx.h"

#include <initializer_list>
#include <string>
#include <vector>

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
 * Extracts integer payload values from a :cpp:class:`TensorProto`.
 *
 * The function first reads from the type-specific repeated fields
 * (``int64_data``, ``int32_data``, ``uint64_data``) when present, and
 * otherwise falls back to decoding ``raw_data`` in little-endian order,
 * as required by ONNX.
 *
 * Supported element types are INT8/16/32/64 and UINT8/16/32/64.
 *
 * @param tensor_proto Tensor to read integer payload values from.
 * @param out Output vector receiving extracted values in storage order.
 *            Cleared before being filled.
 * @return ``true`` on successful extraction, ``false`` when tensor data
 *         is absent or the tensor type/encoding is not supported.
 */
bool ReadIntegerValues(const TensorProto &tensor_proto, std::vector<int64_t> &out);

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

////////////////////
// Node factory
////////////////////

/**
 * Builds a :class:`NodeProto` with the given ``op_type``, input and output
 * names, and optional ``domain`` / ``name``. This is the C++ counterpart to
 * :func:`onnx.helper.make_node` and the recommended way to create a node
 * everywhere a single-node proto is needed (test cases, fixtures, fuzzers,
 * shape-inference unit tests, etc.).
 *
 * Inputs and outputs are passed as ``std::vector<std::string>``, which also
 * accepts brace-enclosed lists of string literals
 * (e.g. ``MakeNode("Add", {"a", "b"}, {"c"})``).
 *
 * @param  op_type Operator type (e.g. ``"Conv"``).
 * @param  inputs  Input names, appended in order.
 * @param  outputs Output names, appended in order.
 * @param  domain  Optional operator domain. When ``nullptr`` the field is
 *                 left untouched (i.e. defaults to the empty ``ai.onnx``
 *                 domain).
 * @param  name    Optional node name. When ``nullptr`` the field is left
 *                 untouched.
 * @return A populated :class:`NodeProto`.
 */
NodeProto MakeNode(const char *op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const char *domain = nullptr,
                   const char *name = nullptr);

/**
 * Appends a new node to ``graph`` with the given ``op_type``, input and output
 * names, and optional ``domain`` / ``name``, and returns a reference to the
 * newly added node. This is a thin convenience wrapper combining
 * :ref:`MakeNode` with ``graph.add_node()``; use it instead of
 * ``*graph.add_node() = MakeNode(...)`` when subsequent code needs to attach
 * attributes to the node.
 *
 * @param  graph   Graph to append the node to.
 * @param  op_type Operator type (e.g. ``"Conv"``).
 * @param  inputs  Input names, appended in order.
 * @param  outputs Output names, appended in order.
 * @param  domain  Optional operator domain.
 * @param  name    Optional node name.
 * @return Reference to the newly added node, owned by ``graph``.
 */
NodeProto &AddNode(GraphProto &graph, const char *op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const char *domain = nullptr,
                   const char *name = nullptr);

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

/**
 * Appends the canonical ``axis`` INT attribute to ``node``. Shorthand for the
 * ``axis``-INT attribute that virtually every ONNX op exposing an axis uses
 * (``Concat``, ``Softmax``, ``Gather``, ``Split``, ``Cast``, ...). Equivalent
 * to ``AddAttribute<int64_t>(node, "axis", axis)``.
 *
 * @param node Target node.
 * @param axis Axis value (may be negative; the caller is responsible for
 *             passing a valid range for the target op).
 */
inline void AddAxisAttribute(NodeProto &node, int64_t axis) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(axis);
}

/////////////
// Attribute
/////////////

/**
 * Appends an AttributeProto carrying ``value`` named ``name`` to ``node`` and
 * returns a pointer to the newly added attribute.  The proto field used and
 * the recorded ``AttributeProto::AttributeType`` are inferred from ``T``.
 * Specializations are provided for the most common attribute payloads
 * (``int64_t``, ``float``, strings, and homogeneous vectors thereof).
 *
 * @tparam T Attribute value type.
 * @param node Target node.
 * @param name Attribute name.
 * @param value Attribute value.
 * @return Pointer to the newly added attribute.
 */
template <typename T>
AttributeProto *AddAttribute(NodeProto &node, const char *name, const T &value);

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name, const int64_t &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
  return attr;
}

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name, const float &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(value);
  return attr;
}

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name, const std::string &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s(value);
  return attr;
}

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name,
                                    const std::vector<int64_t> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : values) {
    attr->ints().push_back(v);
  }
  return attr;
}

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name,
                                    const std::vector<float> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOATS);
  for (float v : values) {
    attr->floats().push_back(v);
  }
  return attr;
}

template <>
inline AttributeProto *AddAttribute(NodeProto &node, const char *name,
                                    const std::vector<std::string> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &v : values) {
    *attr->add_strings() = utils::String(v);
  }
  return attr;
}

/**
 * Returns a pointer to the first attribute of ``node`` whose name equals
 * ``name``, or ``nullptr`` when no such attribute exists. The returned
 * pointer is non-owning and remains valid for the lifetime of ``node``.
 *
 * @param node Node to scan.
 * @param name Attribute name to look up (null-terminated C string).
 * @return Pointer to the matching attribute, or ``nullptr`` if absent.
 */
inline const AttributeProto *FindAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      return &attr;
    }
  }
  return nullptr;
}

/**
 * Returns the value of the scalar attribute ``name`` of ``node``, or
 * ``default_value`` when the attribute is absent. The proto accessor used
 * to read the value is inferred from ``T``. Specializations are provided
 * for ``int64_t``, ``float``, and ``std::string``.
 *
 * @tparam T Attribute scalar type.
 * @param node Node to scan.
 * @param name Attribute name.
 * @param default_value Value returned when the attribute is missing.
 */
template <typename T>
T GetAttributeOr(const NodeProto &node, const char *name, const T &default_value);

template <>
inline int64_t GetAttributeOr(const NodeProto &node, const char *name,
                              const int64_t &default_value) {
  const AttributeProto *attr = FindAttribute(node, name);
  return attr == nullptr ? default_value : attr->ref_i();
}

template <>
inline float GetAttributeOr(const NodeProto &node, const char *name, const float &default_value) {
  const AttributeProto *attr = FindAttribute(node, name);
  return attr == nullptr ? default_value : attr->ref_f();
}

template <>
inline std::string GetAttributeOr(const NodeProto &node, const char *name,
                                  const std::string &default_value) {
  const AttributeProto *attr = FindAttribute(node, name);
  return attr == nullptr ? default_value : attr->ref_s().as_string();
}

/**
 * Reads the repeated INTS attribute ``name`` of ``node``. When present its
 * values are appended to ``out`` in order and the function returns ``true``;
 * otherwise ``out`` is left unchanged and the function returns ``false``.
 *
 * @param node Node to scan.
 * @param name Attribute name.
 * @param out  Destination vector. Values are appended (existing content is
 *             preserved).
 * @return ``true`` when the attribute was found, ``false`` otherwise.
 */
inline bool GetAttributeInts(const NodeProto &node, const char *name, std::vector<int64_t> &out) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return false;
  }
  for (int64_t v : attr->ref_ints()) {
    out.push_back(v);
  }
  return true;
}

/**
 * Returns a reference to the GraphProto carried by the attribute named ``attr_name``
 * on ``node``. Throws ``std::invalid_argument`` when the attribute is missing or
 * does not hold a GraphProto. The optional ``context`` string is prefixed to the
 * thrown error message (e.g. the name of the caller) for diagnostic purposes.
 *
 * @param node      Node to inspect.
 * @param attr_name Name of the attribute to look up.
 * @param context   Optional caller context used as a prefix in error messages.
 * @return Const reference to the GraphProto attribute.
 */
const GraphProto &FindGraphAttribute(const NodeProto &node, const char *attr_name,
                                     const char *context = nullptr);

} // namespace ONNX_LIGHT_NAMESPACE
