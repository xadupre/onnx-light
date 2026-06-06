// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds a TfIdfVectorizer NodeProto with the attribute set shared by every
// upstream backend test (``mode="TF"``, integer pool, no ``weights``).
NodeProto MakeTfIdfNode(int64_t min_gram_length, int64_t max_gram_length, int64_t max_skip_count,
                        const std::vector<int64_t> &ngram_counts,
                        const std::vector<int64_t> &ngram_indexes,
                        const std::vector<int64_t> &pool_int64s) {
  NodeProto node;
  node.set_op_type("TfIdfVectorizer");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute(node, "mode", std::string("TF"));
  AddAttribute(node, "min_gram_length", min_gram_length);
  AddAttribute(node, "max_gram_length", max_gram_length);
  AddAttribute(node, "max_skip_count", max_skip_count);
  AddAttribute(node, "ngram_counts", ngram_counts);
  AddAttribute(node, "ngram_indexes", ngram_indexes);
  AddAttribute(node, "pool_int64s", pool_int64s);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// TfIdfVectorizer — extracts n-grams from a ``[C]`` or ``[N, C]`` integer
// (or string) input and produces a ``tensor(float)`` n-gram count vector
// (since opset 9 in the ai.onnx domain). Test cases mirror the upstream
// ``onnx.backend.test.case.node.tfidfvectorizer`` exporters.
// ---------------------------------------------------------------------------
void RegisterTfIdfVectorizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(9);
  const kernel::KernelContext ctx{opset};
  const kernel::TfIdfVectorizer tf_idf{ctx};

  using Mode = kernel::TfIdfVectorizer::Mode;

  // Shared pool / indexing used by most of the upstream cases.
  const std::vector<int64_t> default_pool{2, 3, 5, 4, 5, 6, 7, 8, 6, 7};
  const std::vector<int64_t> default_ngram_counts{0, 4};
  const std::vector<int64_t> default_ngram_indexes{0, 1, 2, 3, 4, 5, 6};

  // tf_only_bigrams_skip0 — 1-D input, only bigrams, no skips.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/2, /*max_gram_length=*/2,
                                   /*max_skip_count=*/0, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {12}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/2, /*max_gram_length=*/2,
                      /*max_skip_count=*/0, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_only_bigrams_skip0", {opset}, "backend-test",
           registry);
  }

  // tf_onlybigrams_levelempty — pool only contains bigrams (empty unigram
  // level). Counts of [5,6], [7,8] and [6,7] are 1.
  {
    const std::vector<int64_t> pool{5, 6, 7, 8, 6, 7};
    const std::vector<int64_t> ngram_counts{0, 0};
    const std::vector<int64_t> ngram_indexes{0, 1, 2};
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/2, /*max_gram_length=*/2,
                                   /*max_skip_count=*/0, ngram_counts, ngram_indexes, pool);
    Tensor x = Tensor::FromInt32("X", {12}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/2, /*max_gram_length=*/2,
                      /*max_skip_count=*/0, ngram_counts, ngram_indexes, pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_onlybigrams_levelempty", {opset},
           "backend-test", registry);
  }

  // tf_onlybigrams_skip5 — 1-D input, bigrams only, max_skip_count=5.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/2, /*max_gram_length=*/2,
                                   /*max_skip_count=*/5, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {12}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/2, /*max_gram_length=*/2,
                      /*max_skip_count=*/5, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_onlybigrams_skip5", {opset}, "backend-test",
           registry);
  }

  // tf_uniandbigrams_skip5 — 1-D input, uni- and bi-grams, skip=5.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/1, /*max_gram_length=*/2,
                                   /*max_skip_count=*/5, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {12}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/1, /*max_gram_length=*/2,
                      /*max_skip_count=*/5, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_uniandbigrams_skip5", {opset},
           "backend-test", registry);
  }

  // tf_batch_onlybigrams_skip0 — 2-D input, only bigrams, no skips.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/2, /*max_gram_length=*/2,
                                   /*max_skip_count=*/0, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {2, 6}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/2, /*max_gram_length=*/2,
                      /*max_skip_count=*/0, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_batch_onlybigrams_skip0", {opset},
           "backend-test", registry);
  }

  // tf_batch_onlybigrams_skip5 — 2-D input, only bigrams, skip=5.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/2, /*max_gram_length=*/2,
                                   /*max_skip_count=*/5, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {2, 6}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/2, /*max_gram_length=*/2,
                      /*max_skip_count=*/5, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_batch_onlybigrams_skip5", {opset},
           "backend-test", registry);
  }

  // tf_batch_uniandbigrams_skip5 — 2-D input, uni- and bi-grams, skip=5.
  {
    NodeProto node = MakeTfIdfNode(/*min_gram_length=*/1, /*max_gram_length=*/2,
                                   /*max_skip_count=*/5, default_ngram_counts,
                                   default_ngram_indexes, default_pool);
    Tensor x = Tensor::FromInt32("X", {2, 6}, {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8});
    Tensor y = tf_idf(x, Mode::kTF, /*min_gram_length=*/1, /*max_gram_length=*/2,
                      /*max_skip_count=*/5, default_ngram_counts, default_ngram_indexes,
                      default_pool, {}, {});
    Expect(node, {x}, {y}, "test_cc_tfidfvectorizer_tf_batch_uniandbigrams_skip5", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
