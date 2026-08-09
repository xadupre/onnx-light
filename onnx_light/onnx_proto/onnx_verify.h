#pragma once

#include "onnx.h"

#include <unordered_set>

/**
 * @file onnx_verify.h
 * @brief Schema-free structural validation for onnx_proto messages.
 *
 * The functions declared here validate that a protobuf is internally
 * consistent on its own terms: required fields are set, names are unique,
 * the graph is in single static assignment (SSA) form and topologically
 * sorted, tensor payload sizes match the declared shape/dtype, and so on.
 * None of these checks needs an operator-schema registry: nothing here
 * depends on the semantics of a specific ``op_type``.
 *
 * For schema-aware validation (operator input/output arity, attribute
 * constraints, type/shape inference), use ``onnx_lib::checker::check_model()``
 * instead (declared in ``onnx_lib/checker.h``).
 *
 * Every function below throws ``std::invalid_argument`` (via
 * ``EXT_THROW_INVALID`` / ``EXT_ENFORCE_INVALID``) with a descriptive message
 * on the first violation found.
 */

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Validates a ValueInfoProto.
 *
 * @param value_info Value information to validate.
 * @param is_main_graph When false (subgraph input/output), the ``type`` field
 * is not required to be present, mirroring the relaxed constraint that
 * applies to control-flow subgraphs.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyValueInfo(const ValueInfoProto &value_info,
                                          bool is_main_graph = true);

/**
 * Validates a TensorProto: data_type is set, at most one payload field is
 * populated, and the populated field matches the declared data_type.
 *
 * @param tensor Tensor to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyTensor(const TensorProto &tensor);

/**
 * Validates a SparseTensorProto: ``values`` and ``indices`` are individually
 * valid tensors and ``indices`` uses the required INT64 data type.
 *
 * @param sparse_tensor Sparse tensor to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifySparseTensor(const SparseTensorProto &sparse_tensor);

/**
 * Validates an AttributeProto: exactly one value field is populated and it
 * matches the declared ``type``; nested graphs/tensors are recursively
 * validated.
 *
 * @param attribute Attribute to validate.
 * @param in_function_body True when the attribute belongs to a node inside a
 * FunctionProto body, in which case ``ref_attr_name`` is permitted.
 * @param scope Names visible to the attribute's nested subgraph (if any),
 * i.e. every name defined so far in the enclosing graph. Used to validate
 * that control-flow body subgraphs (Attribute.g / Attribute.graphs) can
 * legally reference outer-scope values.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyAttribute(const AttributeProto &attribute, bool in_function_body,
                                          const std::unordered_set<std::string> &scope);

/**
 * Validates a NodeProto: ``op_type`` is set, the node has at least one input
 * or output, attribute names are unique, and each attribute is valid.
 *
 * @param node Node to validate.
 * @param in_function_body True when ``node`` belongs to a FunctionProto body.
 * @param scope Names visible to the node, forwarded to VerifyAttribute() for
 * nested subgraph validation.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyNode(const NodeProto &node, bool in_function_body,
                                     const std::unordered_set<std::string> &scope);

/**
 * Validates a GraphProto: inputs/outputs/initializers have unique names, the
 * graph is in SSA form, nodes are topologically sorted (every input is
 * produced by a prior node, initializer, graph input, or outer scope), and
 * every declared output is actually produced.
 *
 * @param graph Graph to validate.
 * @param is_main_graph When false, relaxes the ValueInfoProto ``type``
 * requirement on graph inputs/outputs (subgraphs may omit it).
 * @param in_function_body True when ``graph`` is nested inside a
 * FunctionProto body.
 * @param outer_scope Names visible from an enclosing graph (used when
 * validating a control-flow body subgraph); may be null for the main graph.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyGraph(const GraphProto &graph, bool is_main_graph = true,
                                      bool in_function_body = false,
                                      const std::unordered_set<std::string> *outer_scope = nullptr);

/**
 * Validates a FunctionProto: ``name`` is set, inputs are uniquely named,
 * nodes are topologically sorted with respect to the function's inputs, and
 * every declared output is produced.
 *
 * @param function Function to validate.
 *
 * @throws std::invalid_argument Thrown when validation fails.
 */
ONNX_LIGHT_PROTO_API void VerifyFunction(const FunctionProto &function);

/**
 * Validates an in-memory ModelProto without requiring an operator-schema
 * registry: ``graph`` is present, at least one opset is imported with unique
 * domains, the main graph is structurally valid, and model-local functions
 * are individually valid and uniquely identified by (domain, name, overload).
 *
 * This performs IR-level structural checks only; it does not check operator
 * input/output arity or attribute constraints, and it does not run shape
 * inference. Use ``onnx_lib::checker::check_model()`` for that.
 *
 * @param model Model to validate.
 *
 * @throws std::invalid_argument Thrown when a structural inconsistency is found.
 */
ONNX_LIGHT_PROTO_API void VerifyModel(const ModelProto &model);

} // namespace ONNX_LIGHT_NAMESPACE
