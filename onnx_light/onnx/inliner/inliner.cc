// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/inliner/inliner.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/checker.h"
#include "onnx/common/assertions.h"
#include "onnx/common/constants.h"
#include "onnx/common/proto_util.h"
#include "onnx/common/visitor.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/shape_inference/attribute_binder.h"
#ifdef ONNX_LIGHT_VERSION_CONVERTER
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
#endif

namespace ONNX_LIGHT_NAMESPACE {
namespace inliner {

namespace { // internal/private API

using OpsetMapBase = std::unordered_map<std::string, int64_t>;

// A representation of the opset versions required by a model or a function.
// Used to check for compatibility between a model and a function or between
// two functions.
struct OpsetMap : public OpsetMapBase {
public:
  // Construct a map representing the opset versions required by a model.
  explicit OpsetMap(const ModelProto &model) { (void)Add(model.opset_import()); }

  // Adds the opset versions required by a function to the map. Returns true
  // iff the function is compatible with the map, i.e., if the function does
  // not require a different version for any domain already in the map.
  bool Add(const FunctionProto &function) { return Add(function.opset_import()); }

  // Returns the set of mismatches in the opset requirements of given
  // function and the map.
  OpsetMapBase Mismatches(const FunctionProto &function) const {
    return Mismatches(function.opset_import());
  }

private:
  OpsetMapBase Mismatches(const utils::RepeatedProtoField<OperatorSetIdProto> &list) const {
    OpsetMapBase result;
    for (const auto &pair : list) {
      auto iter = this->find(NormalizeDomain(pair.domain().as_string()));
      if ((iter != this->end()) && (iter->second != pair.version()))
        result.insert(*iter);
    }
    return result;
  }

  bool Add(const utils::RepeatedProtoField<OperatorSetIdProto> &list) {
    for (const auto &pair : list) {
      auto domain = NormalizeDomain(pair.domain().as_string());
      auto version = pair.version();
      auto iter = this->find(domain);
      if (iter != this->end()) {
        if (iter->second != version)
          return false;
      } else {
        (*this)[domain] = version;
      }
    }
    return true;
  }
};

using RepeatedNodeProto = utils::RepeatedProtoField<NodeProto>;

class NameGenerator : private internal::Visitor {
public:
  explicit NameGenerator(const GraphProto &graph) : index_(0) { NameGenerator::VisitGraph(graph); }

  explicit NameGenerator(const FunctionProto &function) : index_(0) {
    NameGenerator::VisitFunction(function);
  }

  void ResetFor(const GraphProto &graph) {
    index_ = 0;
    existing_names_.clear();
    NameGenerator::VisitGraph(graph);
  }

  void ResetFor(const FunctionProto &function) {
    index_ = 0;
    existing_names_.clear();
    NameGenerator::VisitFunction(function);
  }

  // Creates a new unique name, based on a suggested name, and adds it to the set
  // of existing names. Returns the newly created name.
  std::string CreateNew(const std::string &suggested) {
    std::string name = suggested;
    while (existing_names_.count(name) > 0) {
      name = suggested + "_" + std::to_string(index_++);
    }
    existing_names_.insert(name);
    return name;
  }

  void Add(const std::string &name) {
    // We don't bother to check for empty string names. Ok to add them.
    existing_names_.insert(name);
  }

  bool ProcessGraph(const GraphProto &graph) override {
    for (const auto &x : graph.input())
      Add(x.name().as_string());
    for (const auto &x : graph.initializer())
      Add(x.name().as_string());
    // Adding graph outputs is redundant for a valid graph, but we do it anyway,
    // to produce better results for invalid graphs.
    for (const auto &x : graph.output())
      Add(x.name().as_string());
    return true;
  }

  bool ProcessFunction(const FunctionProto &function) override {
    for (const auto &x : function.input())
      Add(x.as_string());
    for (const auto &x : function.output())
      Add(x.as_string());
    return true;
  }

  bool ProcessNode(const NodeProto &node) override {
    // We use a single name-space for node names and variable names, to keep name-generation simple.
    Add(node.name().as_string());
    for (const auto &name : node.input()) {
      Add(name.as_string());
    }
    for (const auto &name : node.output()) {
      Add(name.as_string());
    }
    return true;
  }

private:
  unsigned int index_;
  std::unordered_set<std::string> existing_names_;
};

class InliningRenamer : public internal::MutableVisitor {
private:
  std::string suffix;
  NameGenerator &generator;

protected:
  std::vector<std::unordered_map<std::string, std::string>> rename_scopes;

public:
  InliningRenamer(std::string suffix_, NameGenerator &generator_)
      : suffix(std::move(suffix_)), generator(generator_) {
    // Create an empty mapping for the top-level scope.
    rename_scopes.emplace_back();
  }

  // We use a two-level renaming scheme to generate names for variables when inlined in the
  // main graph. First, we add a suffix (specific to the call-site being inlined).
  // Thus, "temp" in called-function becomes "temp__1" for the first inlined function-call
  // and "temp__2" for the second inlined function-call. In addition, there is a subsequent
  // iterative check that ensures that this names does not clash with any pre-existing names,
  // and tries another counter-based suffix in the case of a clash, stopping when successful.
  std::string MakeUnique(const std::string &name) { return generator.CreateNew(name + suffix); }

  /**
   * @brief Binds a formal parameter name to an actual parameter name.
   *
   * @param formal_name The formal parameter name to bind.
   * @param actual_name The actual parameter name to bind to.
   */
  void BindFormalToActual(const std::string &formal_name, const std::string &actual_name) {
    auto &current_scope = rename_scopes.back();
    current_scope[formal_name] = actual_name;
  }

  /**
   * @brief Creates a unique name for the given name and binds it.
   *
   * This method creates a unique name based on the suffix and binds the original
   * name to the unique name for later reference renaming.
   *
   * @param original_name The name to create a unique version of.
   * @return The unique name that was created and bound.
   */
  std::string BindToUniqueName(const std::string &original_name) {
    // First create the unique name using MakeUnique
    std::string unique_name = MakeUnique(original_name);

    // Then bind the original name to the unique name
    auto &current_scope = rename_scopes.back();
    current_scope[original_name] = unique_name;

    return unique_name;
  }

  template <bool isOutput>
  void Bind(utils::RepeatedField<utils::String> &formals,
            const utils::RepeatedField<utils::String> &actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name
    // AP. However, if AP is empty, it is a missing optional parameter. This does not make any
    // difference for inputs. However, for outputs we use a unique dummy name to handle the case
    // that it is used in an output-context where it is not optional.
    ONNX_ASSERTM(actuals.size() <= formals.size(),
                 "Number of actual parameters cannot exceed number of formal parameters")
    auto &current_scope = rename_scopes.back();
    size_t i = 0;
    for (; i < actuals.size(); ++i) {
      utils::String &formal = formals[i];
      std::string rename_as = actuals[i].as_string();
      if (isOutput)
        if (rename_as.empty())
          rename_as = MakeUnique(formal.as_string());
      current_scope[formal.as_string()] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
    for (; i < formals.size(); ++i) {
      utils::String &formal = formals[i];
      std::string rename_as = isOutput ? MakeUnique(formal.as_string()) : std::string();
      current_scope[formal.as_string()] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
  }

  // Process a node:
  bool ProcessNode(NodeProto *node) override {
    if (!node->name().empty())
      node->set_name(MakeUnique(node->name().as_string()));

    for (auto &x : node->input()) {
      std::string s = x.as_string();
      LookupOrRename(s, false);
      x = s;
    }
    for (auto &y : node->output()) {
      std::string s = y.as_string();
      LookupOrRename(s, true);
      y = s;
    }
    return true; // Process attribute subgraphs in traversal
  }

  // Process a sub-graph, contained as an attribute in a control-flow op node.
  // Since we need both pre-processing and post-processing in the traversal, we
  // override the VisitGraph method.
  void VisitGraph(GraphProto *graph) override {
    rename_scopes.emplace_back();
    for (auto &x : graph->input()) {
      std::string s = x.name().as_string();
      Rename(s);
      x.set_name(s);
    }
    for (auto &init : graph->initializer()) {
      std::string s = init.name().as_string();
      Rename(s);
      init.set_name(s);
    }
    for (auto &y : graph->output()) {
      std::string s = y.name().as_string();
      Rename(s);
      y.set_name(s);
    }
    for (auto &n : graph->node())
      VisitNode(&n);
    rename_scopes.pop_back();
  }

private:
  // Replace given name with a unique version of the name, and cache the
  // renaming-binding in current scope.
  void Rename(std::string &name) {
    auto new_name = MakeUnique(name);
    auto &current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  void LookupOrRename(std::string &name, bool is_new_def) {
    if (name.empty())
      return;
    for (auto i = rename_scopes.size(); i > 0; --i) {
      const auto &map = rename_scopes[i - 1];
      auto iter = map.find(name);
      if (iter != map.end()) {
        name = iter->second;
        return;
      }
    }
    if (is_new_def) {
      Rename(name);
    }
    // Otherwise, it is a reference to an outer-scope variable that should not be renamed.
  }

public:
  // Renames variables in a FunctionProto for inlining a particular call-site. This does the
  // following: (i)  Rename all intermediate variables in the function to ensure that they are
  // unique (wrt the main graph). (ii) Rename inputs and outputs using names of actual parameters.
  static void Rename(const NodeProto &callnode, FunctionProto &callee, std::string unique_suffix,
                     NameGenerator &generator) {
    InliningRenamer renamer(std::move(unique_suffix), generator);

    renamer.Bind<false>(callee.input(), callnode.input());
    renamer.Bind<true>(callee.output(), callnode.output());

    renamer.VisitFunction(&callee);
    for (auto &v : callee.value_info()) {
      std::string name_str = v.name().as_string();
      renamer.LookupOrRename(name_str, false);
      v.set_name(name_str);
    }
  }
};

// Identify the set of all "input" variables used by a given node.
// This includes the variables listed as node.input, as well as
// implicit inputs referred to in any graph-valued-attribute of the node.
// In the case of variables referenced in sub-graphs, only non-local variables
// are treated as implicit inputs.

class ComputeInputs : private internal::Visitor {
private:
  std::vector<std::unordered_set<std::string>> namescopes;

  bool InNestedScope() const { return !namescopes.empty(); }

  std::unordered_set<std::string> &CurrentScope() { return namescopes.back(); }

  bool IsLocalVar(const std::string &name) const {
    for (const auto &scope : namescopes) {
      if (scope.count(name) > 0) {
        return true;
      }
    }
    return false;
  }

  void VisitGraph(const GraphProto &graph) override {
    namescopes.emplace_back();
    for (const auto &x : graph.input())
      CurrentScope().insert(x.name().as_string());
    for (const auto &init : graph.initializer())
      CurrentScope().insert(init.name().as_string());
    for (const auto &n : graph.node())
      VisitNode(n);
    namescopes.pop_back();
  }

  bool ProcessNode(const NodeProto &node) override {
    for (const auto &var : node.input()) {
      if (!var.empty() && !IsLocalVar(var.as_string())) {
        result.push_back(var.as_string());
      }
    }
    if (InNestedScope()) {
      for (const auto &var : node.output()) {
        if (!var.empty()) {
          CurrentScope().insert(var.as_string());
        }
      }
    }
    return true; // process sub-graphs
  }

public:
  std::vector<std::string> result;

  explicit ComputeInputs(const NodeProto &node) {
    result.reserve(node.input().size());
    ComputeInputs::VisitNode(node);
  }
};

std::vector<std::string> GetUsedVars(const NodeProto &node) { return ComputeInputs(node).result; }

using ConstNodeMap = std::unordered_map<std::string, const NodeProto *>;

ConstNodeMap FindConstantNodes(const GraphProto &graph) {
  ConstNodeMap result;
  for (const NodeProto &node : graph.node()) {
    if (IsOnnxDomain(node.domain().as_string()) && (node.op_type() == "Constant")) {
      result[node.output()[0].as_string()] = &node;
    }
  }
  return result;
}

const TypeProto &GetType(const ModelProto &model, const std::string &var) {
  for (const auto &vi : model.graph().value_info()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (const auto &vi : model.graph().input()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (const auto &vi : model.graph().output()) {
    if (vi.name() == var)
      return vi.type();
  }
  ONNX_ASSERTM(false, "Type unknown for %s", var.c_str())
}

#ifdef ONNX_LIGHT_VERSION_CONVERTER
void ConvertVersion(ModelProto &model, const NodeProto &call_node, FunctionProto &function,
                    int target_version) {
  shape_inference::InferShapes(model);

  ModelProto function_as_model;
  function_as_model.set_ir_version(model.ir_version());
  function_as_model.opset_import().extend(function.opset_import());

  GraphProto &graph = function_as_model.graph();
  // The graph's inputs are all the variables used in the call_node.
  auto used_vars = GetUsedVars(call_node);
  auto constant_node_map = FindConstantNodes(model.graph());

  RepeatedNodeProto &function_nodes = function.node();
  RepeatedNodeProto &nodes = graph.node();
  nodes.reserve(function_nodes.size() + used_vars.size());

  for (const auto &var : used_vars) {
    auto *new_input = graph.add_input();
    new_input->set_name(var);
    new_input->set_type(GetType(model, var));
    // Create a copy of constants used by the call_node.
    // We do not handle initializers-as-constants for now.
    auto it = constant_node_map.find(var);
    if (it != constant_node_map.end()) {
      nodes.push_back(*(it->second));
    }
  }

  // outputs: from call_node node outputs
  for (const auto &var : call_node.output()) {
    if (!var.empty()) {
      auto *new_output = graph.add_output();
      new_output->set_name(var.as_string());
      new_output->set_type(GetType(model, var.as_string()));
    }
  }

  for (auto &function_node : function_nodes)
    nodes.push_back(function_node);
  function_nodes.clear();

  auto converted =
      ONNX_LIGHT_NAMESPACE::version_conversion::ConvertVersion(function_as_model, target_version);

  std::swap(function_nodes, converted.graph().node());

  // Append new initializers to main graph initializers
  for (const auto &added_initializer : converted.graph().initializer())
    model.graph().add_initializer(added_initializer);
  for (const auto &added_initializer : converted.graph().sparse_initializer())
    model.graph().add_sparse_initializer(added_initializer);
}
#endif // ONNX_LIGHT_VERSION_CONVERTER

int64_t GetDomainVersion(const ModelProto &model, const std::string &domain) {
  for (const auto &opset : model.opset_import()) {
    if (opset.domain() == domain) {
      return opset.version();
    }
  }
  return 0;
}

class VectorSet : public FunctionIdSet {
public:
  VectorSet(FunctionIdVector &&function_ids, bool invert)
      : function_ids_(std::move(function_ids)), invert_(invert) {}

  bool Contains(const std::string &function_domain,
                const std::string &function_name) const override {
    bool found = std::find(function_ids_.begin(), function_ids_.end(),
                           std::make_pair(function_domain, function_name)) != function_ids_.end();
    return invert_ ? !found : found;
  }

private:
  FunctionIdVector function_ids_;
  bool invert_;
};

constexpr int64_t kNoConversion = -1;
using FunctionMap = std::unordered_map<FunctionImplId, std::pair<const FunctionProto *, int64_t>>;

using NodeList = utils::RepeatedProtoField<NodeProto>;
using InlinerValueInfoList = utils::RepeatedProtoField<ValueInfoProto>;

struct InlinerImpl {
  ModelProto &model;
  const FunctionIdSet &to_inline;
  const FunctionMap *function_map;
  const ISchemaRegistry *schema_registry = nullptr;
  NameGenerator name_generator;
  int inline_count = 0;

  // Construct inliner for inlining call-sites inside main graph of a model.
  InlinerImpl(ModelProto &model_, const FunctionIdSet &to_inline_, const FunctionMap *function_map_,
              const ISchemaRegistry *schema_registry_)
      : model(model_), to_inline(to_inline_), function_map(function_map_),
        schema_registry(schema_registry_), name_generator(model_.graph()) {}

  virtual ~InlinerImpl() = default;

  virtual bool GetCallee(const NodeProto &node, FunctionProto &callee, int64_t &target_version) {
    std::string domain = node.domain().as_string();
    std::string function_name = node.op_type().as_string();
    if (!to_inline.Contains(domain, function_name)) {
      return false;
    }

    if (function_map != nullptr) {
      if (auto iter = this->function_map->find(GetCalleeId(node));
          iter != this->function_map->end()) {
        auto &[func_ptr, version] = iter->second;
        callee.CopyFrom(*func_ptr);
        target_version = version;
        return true;
      }
    }
    if (schema_registry != nullptr) {
      int64_t domain_version = GetDomainVersion(model, domain);
      const auto *const op_schema =
          schema_registry->GetSchema(node.op_type().as_string(), domain_version, domain);

      if (op_schema == nullptr) {
        // If the schema is not found, we cannot inline the function.
        return false;
      }

      if (op_schema->HasFunction()) {
        const FunctionProto *function_ptr = op_schema->GetFunction(domain_version, false);
        if (function_ptr != nullptr) {
          callee.CopyFrom(*function_ptr);
          target_version = kNoConversion;
          return true;
        }
      }

      // Check if this node has a schema defined function proto.
      if (op_schema->HasContextDependentFunction()) {
#ifdef ONNX_LIGHT_VERSION_CONVERTER
        shape_inference::InferShapes(model); // TODO(ONNX): do shape inference incrementally
#endif
        std::vector<TypeProto> input_types;
        for (const auto &input : node.input()) {
          input_types.emplace_back(GetType(model, input.as_string()));
        }
        ONNX_LIGHT_NAMESPACE::FunctionBodyBuildContextImpl function_body_ctx(node, input_types);
        target_version = kNoConversion;
        // BuildContextDependentFunction is void in onnx-light; it throws on failure.
        op_schema->BuildContextDependentFunction(function_body_ctx, callee, domain_version);
        return true;
      }
    }
    return false;
  }

  /** Shared utility function used for inlining into either a GraphProto or a FunctionProto.
   * @param nodes Mutable list of nodes (of function or graph)
   * @param value_infos Mutable list of value_infos (of function or graph)
   */
  void Process(NodeList &nodes, InlinerValueInfoList &value_infos) {
    NodeList original_nodes;
    // Move all nodes into original_nodes
    std::swap(original_nodes, nodes);

    std::function<void(NodeProto & node)> append_node = [&](NodeProto &node) {
      FunctionProto callee;
      int64_t target_version = kNoConversion;
      if (GetCallee(node, callee, target_version)) {
        // Bind attribute parameters
        internal::AttributeBinder::BindAttributes(node, callee);

        // Rename variable names in callee
        InliningRenamer::Rename(node, callee, "__" + std::to_string(++(this->inline_count)),
                                this->name_generator);
        if (target_version != kNoConversion) {
#ifdef ONNX_LIGHT_VERSION_CONVERTER
          ConvertVersion(model, node, callee, target_version);
#endif
        }
        std::unordered_set<std::string> actual_parameters;
        for (const auto &x : node.input())
          actual_parameters.insert(x.as_string());
        for (const auto &x : node.output())
          actual_parameters.insert(x.as_string());
        // Append valueinfos of called function
        for (const auto &callee_vi : callee.value_info()) {
          if (actual_parameters.count(callee_vi.name().as_string()) == 0) {
            value_infos.push_back(callee_vi);
          }
        }
        // Append nodes of called function
        for (auto &callee_node : callee.node())
          append_node(callee_node);
      } else {
        // Append node without inlining.
        for (auto &attr : node.attribute()) {
          if (attr.has_g()) {
            ProcessGraph(attr.g());
          }
          for (auto &g : attr.graphs()) {
            ProcessGraph(g);
          }
        }

        nodes.push_back(node);
      }
    };
    for (auto &node : original_nodes) {
      append_node(node);
    }
  }

  /** Utility function used for inlining into a GraphProto.
   * @param graph Mutable graph
   */
  void ProcessGraph(GraphProto &graph) {
    auto &nodes = graph.node();
    auto &value_infos = graph.value_info();
    Process(nodes, value_infos);
  }

  /** Utility function used for inlining into a FunctionProto.
   * @param function Mutable function
   */
  void ProcessFunction(FunctionProto &function) {
    auto &nodes = function.node();
    auto &value_infos = function.value_info();
    Process(nodes, value_infos);
  }

  static void InlineLocalFunctions(ModelProto &model, bool convert_version) {
    FunctionIdVector empty_set;
    VectorSet all_functions(std::move(empty_set), true);
    OpsetMap model_imports(model);
    FunctionMap map;

    // For every function, we check if there is a mismatch between the opset versions
    // required for the function and the model. If there is no mismatch, we can inline
    // this function. If there is a mismatch only for the standard ONNX domain, we
    // can inline after version-conversion (if the version-conversion is successful).
    // Otherwise, we cannot inline, since currently version-conversion supports only
    // standard ONNX domain.

    for (const auto &function : model.functions()) {
      auto mismatches = model_imports.Mismatches(function);
      auto iter = mismatches.find(ONNX_DOMAIN);
      int64_t target_onnx_version = kNoConversion;
      if (convert_version && (iter != mismatches.end())) {
        target_onnx_version = iter->second;
        mismatches.erase(iter);
      }
      if (mismatches.empty()) {
        map[GetFunctionImplId(function)] =
            std::pair<const FunctionProto *, int64_t>(&function, target_onnx_version);
      }
    }

    InlinerImpl inliner(model, all_functions, &map, nullptr);
    inliner.ProcessGraph(model.graph());

    // Remove all model-local functions. We do not remove functions with a mis-matched
    // opset version. They need to be handled some other way, eg., using a version-adapter.
    auto &local_functions = model.functions();
    for (auto it = local_functions.begin(); it != local_functions.end();) {
      if (map.count(GetFunctionImplId(*it)) > 0)
        it = local_functions.erase(it);
      else
        ++it;
    }
  }

  static void InlineSelectedFunctions(ModelProto &model, const FunctionIdSet &to_inline,
                                      const ISchemaRegistry *schema_registry) {
    OpsetMap model_imports(model);
    FunctionMap map;
    std::vector<FunctionProto *> non_inlined_functions;

    // If there is any mismatch between the opset versions required for any of the
    // functions and the model, the inliner will fail.

    for (auto &function : model.functions()) {
      if (!model_imports.Add(function))
        ONNX_THROW("Model has functions with incompatible opset versions.");
      if (to_inline.Contains(function.domain().as_string(), function.name().as_string())) {
        map[GetFunctionImplId(function)] =
            std::pair<const FunctionProto *, int64_t>(&function, kNoConversion);
      } else {
        non_inlined_functions.push_back(&function);
      }
    }

    InlinerImpl inliner(model, to_inline, &map, schema_registry);
    inliner.ProcessGraph(model.graph());

    for (auto *function_ptr : non_inlined_functions) {
      inliner.ProcessFunction(*function_ptr);
    }

    // Remove all inlined model-local functions.
    auto &local_functions = model.functions();
    for (auto it = local_functions.begin(); it != local_functions.end();) {
      if (map.count(GetFunctionImplId(*it)) > 0)
        it = local_functions.erase(it);
      else
        ++it;
    }
  }

  static void InlineSelectedLocalFunctions(ModelProto &model, const FunctionIdSet &to_inline) {
    InlineSelectedFunctions(model, to_inline, nullptr);
  }
};

} // namespace

// Public API implementation:

std::unique_ptr<FunctionIdSet>
ONNX_LIGHT_NAMESPACE::inliner::FunctionIdSet::Create(FunctionIdVector &&function_ids, bool invert) {
  return std::make_unique<VectorSet>(std::move(function_ids), invert);
}

void InlineLocalFunctions(ModelProto &model, bool convert_version) {
  InlinerImpl::InlineLocalFunctions(model, convert_version);
}

void InlineSelectedLocalFunctions(ModelProto &model, const FunctionIdSet &to_inline) {
  InlinerImpl::InlineSelectedLocalFunctions(model, to_inline);
}

void InlineSelectedFunctions(ModelProto &model, const FunctionIdSet &to_inline) {
  InlineSelectedLocalFunctions(model, to_inline);
}

void InlineSelectedFunctions(ModelProto &model, const FunctionIdSet &to_inline,
                             const ISchemaRegistry *schema_registry) {
  if (schema_registry == nullptr) {
    schema_registry = OpSchemaRegistry::Instance();
  }
  InlinerImpl::InlineSelectedFunctions(model, to_inline, schema_registry);
}

namespace {

using CycleFuncPtr = const FunctionProto *;
using CycleCallGraph = std::unordered_map<CycleFuncPtr, std::unordered_set<CycleFuncPtr>>;

constexpr int kMaxLocalFunctions = 10000;
constexpr size_t kMaxCallDepth = 100;

enum class CycleVisitState : uint8_t { Unvisited, InPath, Done };

void DetectCycleDFS(CycleFuncPtr root, const CycleCallGraph &call_graph,
                    std::unordered_map<CycleFuncPtr, CycleVisitState> &state,
                    std::vector<CycleFuncPtr> &path) {
  using Iter = std::unordered_set<CycleFuncPtr>::const_iterator;
  struct Frame {
    CycleFuncPtr func;
    Iter cur;
    Iter end;
  };
  std::vector<Frame> stack;
  auto push = [&](CycleFuncPtr func) {
    state[func] = CycleVisitState::InPath;
    path.push_back(func);
    if (path.size() > kMaxCallDepth) {
      ONNX_THROW_EX(checker::ValidationError(
          MakeString("Function call chain depth exceeds limit (", kMaxCallDepth,
                     "). The model may be malformed or malicious.")));
    }
    auto it = call_graph.find(func);
    if (it != call_graph.end()) {
      stack.push_back({func, it->second.begin(), it->second.end()});
    } else {
      stack.push_back({func, {}, {}});
    }
  };
  push(root);
  while (!stack.empty()) {
    auto &frame = stack.back();
    if (frame.cur == frame.end) {
      path.pop_back();
      state[frame.func] = CycleVisitState::Done;
      stack.pop_back();
      continue;
    }
    CycleFuncPtr callee = *frame.cur;
    ++frame.cur;
    auto s = state[callee];
    if (s == CycleVisitState::InPath) {
      auto start = std::find(path.begin(), path.end(), callee);
      std::string cycle;
      for (auto cit = start; cit != path.end(); ++cit)
        cycle += (cit == start ? "" : " -> ") + GetFunctionImplId(**cit);
      ONNX_THROW_EX(checker::ValidationError(
          MakeString("Cycle detected in model-local function references: ", cycle, " -> ",
                     GetFunctionImplId(*callee),
                     ". Self-referencing or cyclically-referencing functions would cause infinite "
                     "recursion.")));
    } else if (s == CycleVisitState::Unvisited) {
      push(callee);
    }
  }
}

} // namespace

void CheckFunctionCallCycles(const ModelProto &model) {
  if (model.functions().size() > static_cast<size_t>(kMaxLocalFunctions)) {
    ONNX_THROW_EX(checker::ValidationError(MakeString(
        "Model contains ", model.functions().size(), " local functions, exceeding the limit of ",
        kMaxLocalFunctions, ". The model may be malformed or malicious.")));
  }
  std::unordered_map<std::string, CycleFuncPtr> func_by_key;
  for (const auto &f : model.functions()) {
    const auto function_impl_id = GetFunctionImplId(f);
    const bool inserted = func_by_key.emplace(function_impl_id, &f).second;
    if (!inserted) {
      ONNX_THROW_EX(checker::ValidationError(
          MakeString("Model contains multiple local functions with the same implementation id '",
                     function_impl_id, "'.")));
    }
  }
  CycleCallGraph call_graph;
  for (const auto &entry : func_by_key) {
    const auto *func = entry.second;
    auto &callees = call_graph[func];
    for (const auto &node : func->node()) {
      auto it = func_by_key.find(GetCalleeId(node));
      if (it != func_by_key.end()) {
        callees.insert(it->second);
      }
    }
  }
  std::unordered_map<CycleFuncPtr, CycleVisitState> visit_state;
  std::vector<CycleFuncPtr> path;
  for (const auto &entry : func_by_key) {
    if (visit_state[entry.second] == CycleVisitState::Unvisited) {
      DetectCycleDFS(entry.second, call_graph, visit_state, path);
    }
  }
}

// Implementation of the Renamer class using InliningRenamer directly
class Renamer::Impl {
private:
  NameGenerator generator_;
  InliningRenamer renamer_;

public:
  Impl(const std::string &prefix, const GraphProto &graph)
      : generator_(graph), renamer_("__" + prefix, generator_) {}

  Impl(const std::string &prefix, const FunctionProto &function)
      : generator_(function), renamer_("__" + prefix, generator_) {}

  InliningRenamer &GetRenamer() { return renamer_; }

  void BindName(const std::string &formal_name, const std::string &actual_name) {
    renamer_.BindFormalToActual(formal_name, actual_name);
  }

  void RenameNode(NodeProto &node) {
    // Use the InliningRenamer's ProcessNode method which handles graph-value attributes
    renamer_.ProcessNode(&node);
  }
};

Renamer::Renamer(const std::string &prefix, const GraphProto &graph)
    : pImpl_(std::make_unique<Impl>(prefix, graph)) {}

Renamer::Renamer(const std::string &prefix, const FunctionProto &function)
    : pImpl_(std::make_unique<Impl>(prefix, function)) {}

Renamer::~Renamer() = default;

void Renamer::BindName(const std::string &formal_name, const std::string &actual_name) {
  pImpl_->BindName(formal_name, actual_name);
}

void Renamer::RenameNode(NodeProto &node) { pImpl_->RenameNode(node); }

std::string Renamer::BindToUniqueName(const std::string &original_name) {
  return pImpl_->GetRenamer().BindToUniqueName(original_name);
}

} // namespace inliner

// FunctionBuilder::AddInlinedCall is defined here (not in function.cc) because
// it depends on inliner::Renamer.
FunctionBuilder &FunctionBuilder::AddInlinedCall(std::initializer_list<std::string_view> outputs,
                                                 const GraphProto &graph,
                                                 std::initializer_list<std::string_view> inputs,
                                                 std::string_view prefix) {
  // Create a renamer with the given prefix
  inliner::Renamer renamer(std::string(prefix), graph);

  // Bind formal inputs to actual inputs
  const auto *input_it = inputs.begin();
  for (const auto &graph_input : graph.input()) {
    if (input_it != inputs.end()) {
      renamer.BindName(graph_input.name().as_string(), std::string(*input_it));
      ++input_it;
    }
  }

  // Bind formal outputs to actual outputs
  const auto *output_it = outputs.begin();
  for (const auto &graph_output : graph.output()) {
    if (output_it != outputs.end()) {
      renamer.BindName(graph_output.name().as_string(), std::string(*output_it));
      ++output_it;
    }
  }

  // Add Constant nodes for every initializer in the graph
  for (const auto &initializer : graph.initializer()) {
    std::string const_name = renamer.BindToUniqueName(initializer.name().as_string());
    Const(const_name, initializer);
  }

  // Add a copy of every node in the graph with renamed variables
  for (const auto &node : graph.node()) {
    NodeProto new_node;
    new_node.CopyFrom(node);

    // Rename the node using the renamer
    renamer.RenameNode(new_node);

    // Add the node to the function
    funProto.add_node(new_node);
  }

  return *this;
}

} // namespace ONNX_LIGHT_NAMESPACE
