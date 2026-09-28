// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/generation.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/persistent_value_state.h"

#include <cmath>
#include <limits>
#include <random>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

bool HasInput(const ModelProto &model, const std::string &name) {
  return std::any_of(model.graph().input().begin(), model.graph().input().end(),
                     [&](const auto &input) { return input.name() == name; });
}

void ValidateStorage(const Tensor &tensor) {
  const int64_t count = tensor.shape.product();
  EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(tensor.data_type, count) &&
                          (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                      "Generate: invalid tensor storage.");
}

const Tensor &IntMatrix(const RuntimeValueMap &feeds, const std::string &name) {
  const auto it = feeds.find(name);
  EXT_ENFORCE_INVALID(it != feeds.end() && it->second.kind == RuntimeValue::Kind::kTensor,
                      "Generate: missing tensor input '", name, "'.");
  const Tensor &tensor = it->second.tensor;
  EXT_ENFORCE_INVALID(tensor.data_type == DataType::INT64 && tensor.shape.size() == 2 &&
                          tensor.shape[0] > 0 && tensor.shape[1] > 0,
                      "Generate: '", name, "' must be a nonempty INT64 matrix.");
  ValidateStorage(tensor);
  return tensor;
}

std::vector<int64_t> IntValues(const Tensor &tensor) {
  return {tensor.AsInt64(), tensor.AsInt64() + tensor.shape.product()};
}

void AppendColumn(std::vector<int64_t> &matrix, const std::vector<int64_t> &column) {
  const size_t width = matrix.size() / column.size();
  std::vector<int64_t> next;
  next.reserve(matrix.size() + column.size());
  for (size_t row = 0; row < column.size(); ++row) {
    next.insert(next.end(), matrix.begin() + row * width, matrix.begin() + (row + 1) * width);
    next.push_back(column[row]);
  }
  matrix.swap(next);
}

int64_t Sample(const Tensor &logits, int64_t offset, int64_t vocabulary, double temperature,
               std::mt19937_64 &random) {
  std::vector<double> weights(static_cast<size_t>(vocabulary));
  int64_t best = 0;
  for (int64_t token = 0; token < vocabulary; ++token) {
    const double value = logits.data_type == DataType::DOUBLE ? logits.AsDouble()[offset + token]
                                                              : logits.AsFloat()[offset + token];
    EXT_ENFORCE_INVALID(!std::isnan(value) && value != std::numeric_limits<double>::infinity(),
                        "Generate: logits must not contain NaN or positive infinity.");
    weights[token] = value;
    if (value > weights[best])
      best = token;
  }
  const double maximum = weights[best];
  EXT_ENFORCE_INVALID(std::isfinite(maximum), "Generate: all logits are negative infinity.");
  if (temperature == 0.)
    return best;
  for (double &value : weights)
    value = std::exp((value - maximum) / temperature);
  std::discrete_distribution<int64_t> distribution(weights.begin(), weights.end());
  return distribution(random);
}

} // namespace

Tensor Generate(const ModelProto &model, RuntimeContext &context, const RuntimeValueMap &feeds,
                const GenerationOptions &options, RuntimeSessionOptions session_options) {
  EXT_ENFORCE_INVALID(options.max_new_tokens >= 0, "Generate: max_new_tokens must be nonnegative.");
  EXT_ENFORCE_INVALID(std::isfinite(options.temperature) && options.temperature >= 0.,
                      "Generate: temperature must be finite and nonnegative.");
  EXT_ENFORCE_INVALID(!options.eos_token_id || *options.eos_token_id >= 0,
                      "Generate: eos_token_id must be nonnegative.");
  EXT_ENFORCE_INVALID(!options.pad_token_id || *options.pad_token_id >= 0,
                      "Generate: pad_token_id must be nonnegative.");
  EXT_ENFORCE_INVALID(HasInput(model, options.input_ids_name),
                      "Generate: input_ids must name a graph input.");
  EXT_ENFORCE_INVALID(
      std::any_of(model.graph().output().begin(), model.graph().output().end(),
                  [&](const auto &output) { return output.name() == options.logits_name; }),
      "Generate: logits must name a graph output.");
  for (const auto &[name, value] : feeds) {
    (void)value;
    EXT_ENFORCE_INVALID(HasInput(model, name), "Generate: unknown feed '", name, "'.");
  }
  const Tensor &prompt = IntMatrix(feeds, options.input_ids_name);
  const int64_t batch = prompt.shape[0], prompt_length = prompt.shape[1];
  EXT_ENFORCE_INVALID(options.max_new_tokens <= std::numeric_limits<int64_t>::max() - prompt_length,
                      "Generate: requested sequence length overflows.");
  const int64_t max_length = prompt_length + options.max_new_tokens;
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(max_length) <=
                              std::numeric_limits<size_t>::max() / sizeof(int64_t) / batch &&
                          max_length <= std::numeric_limits<int64_t>::max() / batch,
                      "Generate: requested output size overflows.");
  std::vector<int64_t> tokens = IntValues(prompt);
  EXT_ENFORCE_INVALID(std::all_of(tokens.begin(), tokens.end(), [](auto id) { return id >= 0; }),
                      "Generate: token IDs must be nonnegative.");
  const bool has_mask = HasInput(model, options.attention_mask_name);
  const bool has_positions = HasInput(model, options.position_ids_name);
  EXT_ENFORCE_INVALID((!has_mask || options.attention_mask_name != options.input_ids_name) &&
                          (!has_positions || (options.position_ids_name != options.input_ids_name &&
                                              (!has_mask || options.position_ids_name !=
                                                                options.attention_mask_name))),
                      "Generate: token, mask and position input names must be distinct.");
  std::vector<int64_t> mask(tokens.size(), 1);
  if (has_mask && feeds.contains(options.attention_mask_name)) {
    const auto &tensor = IntMatrix(feeds, options.attention_mask_name);
    EXT_ENFORCE_INVALID(tensor.shape == prompt.shape, "Generate: attention mask shape mismatch.");
    mask = IntValues(tensor);
  }
  std::vector<int64_t> positions(tokens.size());
  for (int64_t row = 0; row < batch; ++row) {
    int64_t position = 0;
    bool started = false;
    for (int64_t col = 0; col < prompt_length; ++col) {
      const auto index = row * prompt_length + col;
      EXT_ENFORCE_INVALID(mask[index] == 0 || mask[index] == 1,
                          "Generate: attention mask must contain zeros and ones.");
      EXT_ENFORCE_INVALID(!started || mask[index] == 1,
                          "Generate: only left-padded prompts are supported.");
      started = started || mask[index] == 1;
      positions[index] = mask[index] ? position++ : 0;
    }
    EXT_ENFORCE_INVALID(started, "Generate: each prompt must contain an unmasked token.");
  }
  if (has_positions && feeds.contains(options.position_ids_name)) {
    const auto &tensor = IntMatrix(feeds, options.position_ids_name);
    EXT_ENFORCE_INVALID(tensor.shape == prompt.shape, "Generate: position IDs shape mismatch.");
    positions = IntValues(tensor);
  }
  for (const auto position : positions)
    EXT_ENFORCE_INVALID(position >= 0 && position <= std::numeric_limits<int64_t>::max() -
                                                         options.max_new_tokens,
                        "Generate: invalid or overflowing position ID.");
  if (options.max_new_tokens == 0)
    return Tensor::From<int64_t>(options.input_ids_name, prompt.shape, tokens);

  RuntimeValueMap current;
  for (const auto &[name, value] : feeds)
    current.emplace(name, value.BorrowView());
  std::unique_ptr<PersistentValueState> state;
  std::unique_ptr<RuntimeSession> session;
  if (!model.graph().persistent_bindings().empty()) {
    RuntimeValueMap initial;
    for (const auto &binding : model.graph().persistent_bindings()) {
      const std::string &name = binding.input_name();
      EXT_ENFORCE_INVALID(name != options.input_ids_name &&
                              (!has_mask || name != options.attention_mask_name) &&
                              (!has_positions || name != options.position_ids_name),
                          "Generate: token, mask and position inputs cannot be persistent.");
      auto value = current.extract(name);
      if (!value.empty())
        initial.insert(std::move(value));
    }
    state = std::make_unique<PersistentValueState>(model, std::move(initial), session_options);
  } else {
    session = std::make_unique<RuntimeSession>(model, session_options);
  }
  std::mt19937_64 random(options.seed ? *options.seed : std::random_device{}());
  std::vector<bool> finished(static_cast<size_t>(batch), false);
  std::vector<int64_t> next(static_cast<size_t>(batch));
  std::vector<int64_t> next_positions(static_cast<size_t>(batch));
  std::vector<int64_t> next_mask(static_cast<size_t>(batch));
  int64_t length = prompt_length;
  for (int64_t step = 0; step < options.max_new_tokens; ++step) {
    const bool decode = state && step > 0;
    const Shape shape{batch, decode ? 1 : length};
    current.insert_or_assign(
        options.input_ids_name,
        RuntimeValue(Tensor::From<int64_t>(options.input_ids_name, shape, decode ? next : tokens)));
    if (has_mask)
      current.insert_or_assign(
          options.attention_mask_name,
          RuntimeValue(Tensor::From<int64_t>(options.attention_mask_name, {batch, length}, mask)));
    if (has_positions)
      current.insert_or_assign(
          options.position_ids_name,
          RuntimeValue(Tensor::From<int64_t>(options.position_ids_name, shape,
                                             decode ? next_positions : positions)));
    Tensor logits;
    if (state) {
      auto outputs = state->Run(context, current);
      auto it = outputs.find(options.logits_name);
      EXT_ENFORCE_INVALID(it != outputs.end() && it->second.kind == RuntimeValue::Kind::kTensor,
                          "Generate: logits output must be a tensor.");
      logits = std::move(it->second.tensor);
    } else {
      auto invocation = context.MakeFunctionContext();
      RegisterModelFunctions(model, invocation);
      for (const auto &[name, value] : current)
        invocation.PutValue(name, value.BorrowView(), RuntimeEventKind::kInput);
      session->Run(invocation);
      EXT_ENFORCE_INVALID(invocation.Has(options.logits_name), "Generate: missing logits tensor.");
      logits = std::move(invocation.Get(options.logits_name));
    }
    EXT_ENFORCE_INVALID(
        (logits.shape.size() == 2 || logits.shape.size() == 3) && logits.shape[0] == batch &&
            logits.shape.back() > 0 && (logits.shape.size() == 2 || logits.shape[1] > 0),
        "Generate: logits must have shape [batch, vocabulary] or [batch, sequence, vocabulary].");
    EXT_ENFORCE_INVALID(logits.data_type == DataType::FLOAT ||
                            logits.data_type == DataType::DOUBLE ||
                            IsHalfPrecision(logits.data_type),
                        "Generate: logits must have a floating-point type.");
    ValidateStorage(logits);
    if (IsHalfPrecision(logits.data_type))
      logits = PromoteToFloat32(logits);
    const int64_t vocabulary = logits.shape.back();
    EXT_ENFORCE_INVALID(!options.eos_token_id || *options.eos_token_id < vocabulary,
                        "Generate: eos_token_id is outside the vocabulary.");
    EXT_ENFORCE_INVALID(!options.pad_token_id || *options.pad_token_id < vocabulary,
                        "Generate: pad_token_id is outside the vocabulary.");
    const int64_t stride = logits.shape.product() / batch;
    for (int64_t row = 0; row < batch; ++row) {
      next_mask[row] = finished[row] ? 0 : 1;
      next[row] = finished[row] ? options.pad_token_id.value_or(*options.eos_token_id)
                                : Sample(logits, (row + 1) * stride - vocabulary, vocabulary,
                                         options.temperature, random);
      next_positions[row] = positions[row * length + length - 1] + 1;
      if (options.eos_token_id && next[row] == *options.eos_token_id)
        finished[row] = true;
    }
    AppendColumn(tokens, next);
    ++length;
    if (std::all_of(finished.begin(), finished.end(), [](bool value) { return value; }) ||
        step + 1 == options.max_new_tokens)
      break;
    AppendColumn(mask, next_mask);
    AppendColumn(positions, next_positions);
  }
  return Tensor::From<int64_t>(options.input_ids_name, {batch, length}, tokens);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
