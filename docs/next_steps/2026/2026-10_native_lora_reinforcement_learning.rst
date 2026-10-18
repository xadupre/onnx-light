.. _l-next-steps-native-lora-reinforcement-learning:

Native LoRA fine-tuning and reinforcement learning
=================================================

:Date: 2026-10-18

**discussion -- not implemented**

Objective
+++++++++

Adapts a small pretrained causal language model through interaction with an
environment, using only onnx-light for forward execution, backward execution
and parameter updates. Starts with CPU FLOAT32, LoRA adapters and REINFORCE
with a baseline; does not require PyTorch, ONNX Runtime, a learned reward model
or a critic. A C++ or Python host loop orchestrates episodes and rewards.

The first deliverable is a reproducible small example, not a general agent
framework. Its environment exposes observations, actions, rewards and episode
termination. Uses a bounded, deterministic task with a verifiable reward before
connecting external tools.

Current foundation and gaps
+++++++++++++++++++++++++++

* ``Generate`` already supports autoregressive decoding, temperature sampling,
  EOS, masks and persistent KV state. Its result is token IDs, not a training
  trajectory with policy log-probabilities.
* ``GradientOfNodes`` and ``GradientOfFunction`` generate symbolic backward
  functions. The default registry includes elementary arithmetic, ``MatMul``,
  normalizations including RMSNorm, and some tensor and reduction operations.
* Native Adam, Adagrad and Momentum kernels exist for FLOAT tensors. They are
  building blocks, not a complete parameter/optimizer lifecycle.
* The Tiny-LLM shape-inference case uses random weights and a simplified
  decoder. It is a useful structural fixture, not a pretrained model or proof
  of training support.

The existing gradient milestone remains the foundation:
:ref:`l-next-steps-gradient`. This proposal extends it to executable Transformer
training and adds LoRA and reinforcement learning.

1. Reliable Transformer differentiation
+++++++++++++++++++++++++++++++++++++++

First establishes numerical correctness independently from reinforcement
learning.

* Fixes batched ``MatMul`` derivatives: transposes only the final two axes,
  handles vector operands and broadcasting, and reduces parameter gradients
  across broadcast batch dimensions.
* Reduces broadcast gradients back to each operand's shape for arithmetic
  operations. Handles reduction axes, ``keepdims`` and the actual element count
  reduced by ``ReduceMean``.
* Adds the rules required by the selected graph: stable ``Softmax`` and
  ``LogSoftmax``, token selection, embeddings, attention and relevant tensor
  transformations. Repeated embedding indices accumulate rather than overwrite
  their contributions. Integer tokens, masks and shape computations are not
  differentiated.
* Supports attention either through a backward rule or an explicit
  differentiable decomposition. Covers causal/padding masks and the selected
  model's head layout; adds RoPE and grouped-query attention when that model
  needs them. Does not infer support from the simplified Tiny-LLM fixture.
* Prunes backward computation to requested trainable parameters. Freezing a
  weight suppresses its gradient, not propagation through its operation to
  earlier adapters. Rejects missing rules on required derivative paths.
* Makes forward activations needed by backward explicit: either retains them
  as declared values or recomputes a documented region. Produces executable
  graph/function boundaries, preserves source opsets and constants, and handles
  all differentiated outputs rather than silently choosing one.

Integrates saved activations with existing execution lifetimes and allocator
ownership. Never overwrites a graph input or reuses an activation while backward
still needs it. Avoids an unrelated scheduler or whole-graph rebuild on each
training iteration.

Acceptance: native forward/backward execution passes finite-difference checks
on small tensors, batched projections, broadcast parameters, repeated indices,
normalization, attention and a residual/SwiGLU decoder block. Includes a
multi-operation example whose backward needs intermediate forward values.

2. LoRA graph transformation
++++++++++++++++++++++++++++

For a projection with ``X[..., in]`` and ``W[in, out]``, injects:

.. code-block:: text

   Y = X @ W + (alpha / r) * ((X @ A) @ B)
   A: [in, r]       B: [r, out]

Keeps the pretrained ``W`` frozen and trains only the selected adapters.
Uses reproducibly initialized small random ``A`` and zero ``B`` so the initial
model matches the base model. The first gradient of ``A`` may be zero under
this initialization; the first update of ``B`` must still be effective.

* Selects projections explicitly by stable graph identity, not substring-only
  matching. Validates rank, dtype, orientation and adapter-name collisions.
* Starts with FLOAT32 ``MatMul`` projections. Normalizes supported ``Gemm``
  layouts explicitly or rejects them; never silently adapts the wrong axes.
* Starts with the LM head to prove the training lifecycle, then adapts attention
  query/value projections to exercise backward through attention. MLP and other
  projections follow through the same selection mechanism.
* Keeps shared base weights shared. Records whether adapters are independent
  or shared at multiple call sites, and accumulates shared adapter gradients.
* Exposes adapter tensors as mutable training inputs/state, not constants that
  folding or immutable-weight preparation can bake into stale results.
* Saves adapters separately with target names, shapes, rank, scaling, dtype and
  a base-model identity. Loading rejects incompatible models and missing or
  unexpected tensors.
* Supports inference with separate adapters first. A later explicit merge
  computes ``W + (alpha / r) * A @ B`` into a new export without changing the
  source weights or applying the merge twice.

Freezing the base does not eliminate all activation memory: earlier adapters
still require gradients through later frozen layers. Quantized-base training
(QLoRA), adapter dropout and mixed precision are deferred.

Acceptance: zero-adapter parity, finite-difference adapter gradients, unchanged
base-weight bytes across updates, shared-use accumulation, checkpoint round-trip
and separate/merged inference equivalence within declared FLOAT32 tolerances.

3. Native training step and state
++++++++++++++++++++++++++++++++

Builds a reusable sequence of native graphs/sessions for forward loss,
backward, gradient accumulation/clipping and optimizer updates. Host code
orchestrates these steps but does not compute tensor derivatives or optimizer
updates with NumPy.

* Makes trainable tensors, frozen tensors, optimizer moments, step counter and
  accumulated gradients explicit. Updates only selected parameters.
* Uses existing Adam or Momentum initially. Adds AdamW only with explicit
  decoupled weight-decay semantics; does not relabel ONNX Adam as AdamW.
* Supports variable sequence lengths, microbatches, global gradient-norm
  clipping and correctly weighted accumulation across valid action tokens.
  Rejects a batch with no trainable tokens and reports non-finite loss or
  gradients before committing an update.
* Reuses compiled/prepared structure where valid, but invalidates preparations
  dependent on changed parameter contents. Generation must observe the new
  adapter version, and old KV caches must not survive a policy update.
* Saves/restores adapters, optimizer state, counters, RNG state and training
  configuration coherently. Resumes at an episode/batch boundary initially,
  without claiming to serialize an arbitrary external environment.

Reuses existing runtime state/ownership mechanisms where their contracts fit.
Does not force weights read by several nodes into an exclusive-consumption
persistent binding. Checkpoints carry versioned metadata and validate tensor
sizes, identities and types before execution.

Acceptance: a small supervised causal-language-model task decreases its native
loss using LoRA alone. A resumed deterministic run matches uninterrupted updates,
and the next generation uses updated parameters without rebuilding the model
for every token.

4. Agent trajectories and policy-gradient loss
+++++++++++++++++++++++++++++++++++++++++++++

An episode records observation/action token IDs, causal context, attention
masks, action-token mask, termination reason, reward and generating policy
version. Prompt, padding, tool responses and other environment observations
contribute context but not action-token loss. EOS is an action token when
sampled. Truncation and genuine termination remain distinguishable.

Samples without differentiating the discrete draw or the environment.
Replays recorded sequences with teacher forcing and no KV cache initially.
Aligns each action token with the preceding position's logits; causal masking
prevents future observations from influencing its probability.

For the initial estimator, uses:

.. math::

   L(\theta) =
   -\frac{1}{N}\sum_{i=1}^{N}
   (R_i-b)\sum_{t \in \mathcal{A}_i}
   \log \pi_\theta(a_{i,t}\mid h_{i,t})

Here ``N`` is the number of episodes, ``A_i`` contains their action positions,
and the baseline ``b`` is fixed from previous batches and updated only after
forming the current advantages. Rewards and advantages are detached inputs.
Keeps this episode-mean objective distinct from optional token-normalized
objectives, which change sequence weighting.

* Starts with categorical sampling at temperature one and no top-k/top-p
  filtering. Greedy decoding is not the exploration policy for this estimator.
* Generates the complete batch with fixed parameters, recomputes its
  log-probabilities under that same policy and performs one update. Does not
  reuse stale trajectories for multiple policy updates without an explicit
  off-policy correction or a different objective.
* Adds an optional entropy bonus after the basic estimator works. A frozen
  reference policy and KL regularization are subsequent stabilization work,
  with the estimator and masking documented explicitly.
* Keeps rewards host-computable for verifiable tasks; a learned reward model
  is optional. PPO/GRPO and a value head are not prerequisites.

Bounds episodes, generated tokens and context length. An eventual tool bridge
uses an explicit allowlist, validates structured actions, limits output sizes
and applies timeouts; sampled text never becomes an unchecked shell command.

Acceptance: checks exact masked losses on hand-computable trajectories, token
alignment and policy-version mismatch rejection. A native one-step update
increases the probability of a positively advantaged action. Across fixed
evaluation seeds, a bounded task shows improved reward over the frozen baseline;
does not require every stochastic update to improve reward.

Implementation sequence
+++++++++++++++++++++++

Each milestone depends on the preceding acceptance gate:

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Order
     - Deliverable
     - Acceptance gate
   * - 1
     - Gradient correctness and forward/backward activation contract
     - Numerical primitive checks and executable multi-operation backward.
   * - 2
     - Transformer gradient coverage and trainable-path pruning
     - Small causal decoder passes numerical checks, including frozen layers.
   * - 3
     - LoRA injection and adapter serialization
     - Zero-adapter parity, correct gradients and unchanged base weights.
   * - 4
     - Native supervised LoRA training and resumable optimizer state
     - Loss decreases; checkpoint resume and generation refresh are correct.
   * - 5
     - Trajectory collection, masked REINFORCE loss and bounded environment
     - Correct one-step policy update and reproducible reward improvement.
   * - 6
     - Pretrained small-model example and resource measurements
     - End-to-end adaptation with documented model/tokenizer revisions,
       license, reward evaluation, throughput and peak memory.

The final example records the actual pretrained graph's operator inventory.
Token IDs suffice for the synthetic milestones. The pretrained example must
also supply a compatible tokenizer, chat template and special-token contract;
it either uses native text operators where supported or includes the missing
local tokenizer integration. It must not hide a PyTorch, Transformers or ONNX
Runtime dependency behind tokenization or model conversion at training time.

Initial scope excludes GPU/distributed training, full-model adaptation,
quantized-base LoRA, PPO/GRPO, arbitrary tool execution and a general agent SDK.
Measures rollout and training costs separately, including activation memory,
optimizer state and copies; LoRA parameter savings alone are not a sufficient
memory benchmark.
