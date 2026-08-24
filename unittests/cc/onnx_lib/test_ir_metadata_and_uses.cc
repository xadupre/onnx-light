// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../common/ir.h"
#include <gtest/gtest.h>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(onnx_ir, CopyMetadataPreservesUnknownRank) {
  Graph graph;
  Value *source = graph.addInput();
  Value *destination = graph.addInput();
  destination->setSizes({Dimension(1)});
  ASSERT_FALSE(source->has_sizes());
  ASSERT_TRUE(destination->has_sizes());

  destination->copyMetadata(source);

  EXPECT_FALSE(destination->has_sizes());
}

TEST(onnx_ir, CopyMetadataCopiesKnownSizes) {
  Graph graph;
  Value *source = graph.addInput();
  source->setSizes({Dimension(2), Dimension(3)});
  Value *destination = graph.addInput();

  destination->copyMetadata(source);

  ASSERT_TRUE(destination->has_sizes());
  ASSERT_EQ(destination->sizes().size(), 2u);
  EXPECT_EQ(destination->sizes()[0].dim, 2);
  EXPECT_EQ(destination->sizes()[1].dim, 3);
}

TEST(onnx_ir, HasUsesInCurrentGraphTracksDirectUses) {
  Graph graph;
  Value *input = graph.addInput();
  Node *producer = graph.create(kNeg, 1);
  producer->addInput(input);
  graph.appendNode(producer);
  Value *output = producer->output();

  EXPECT_FALSE(output->hasUsesInCurrentGraph());
  EXPECT_FALSE(producer->hasUsesInCurrentGraph());

  Node *consumer = graph.create(kNeg, 1);
  consumer->addInput(output);
  graph.appendNode(consumer);

  EXPECT_TRUE(output->hasUsesInCurrentGraph());
  EXPECT_TRUE(producer->hasUsesInCurrentGraph());

  consumer->removeInput(0);
  EXPECT_FALSE(output->hasUsesInCurrentGraph());
  EXPECT_FALSE(producer->hasUsesInCurrentGraph());
}

TEST(onnx_ir, EraseInitializerAcceptsAliasedLongName) {
  Graph graph;
  Tensor initializer;
  initializer.setName("scale_initializer_name_longer_than_sso");
  graph.addInitializerAndCreateValue(initializer);
  ASSERT_EQ(graph.initializers().size(), 1u);
  const std::string &aliased_name = graph.initializers().front().name();

  graph.eraseInitializer(aliased_name);

  EXPECT_TRUE(graph.initializers().empty());
  EXPECT_TRUE(graph.initializer_names().empty());
}
