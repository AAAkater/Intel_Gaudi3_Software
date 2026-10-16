#include <memory>

#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "liveness_analysis.h"
#include "node_factory.h"
#include "tensor.h"
#include "graph_optimizer_test.h"

class LaTest : public GraphOptimizerTest {};

TEST_F(LaTest, liveness_analysis_test)
{
    TSize sizes[] = {1,1,1};
    TSize concatSizes[] = {2,1,1};
    // Create original graph

    // Tensor creation
    TensorPtr tensor_0(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_1(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr tensor_2(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_3(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr tensor_4(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_5(new Tensor(3, sizes, syn_type_fixed));

    TensorVector concatinput;
    concatinput.push_back(tensor_3);
    concatinput.push_back(tensor_5);

    TensorPtr tensor_6(new Tensor(3, concatSizes, syn_type_fixed));
    TensorPtr tensor_7(new Tensor(3, sizes, syn_type_fixed));

    GaudiGraph origGraph;
    unsigned expandDim = 0;
    auto node_0 = NodeFactory::createNode({tensor_0}, {tensor_1}, &expandDim, NodeFactory::memcpyNodeTypeName, "");
    auto node_1 = NodeFactory::createNode({tensor_2}, {tensor_3}, &expandDim, NodeFactory::memcpyNodeTypeName, "");
    auto node_2 = NodeFactory::createNode({tensor_2}, {tensor_5}, &expandDim, NodeFactory::memcpyNodeTypeName, "");

    GraphEditor::addNode(origGraph, node_0);
    GraphEditor::addNode(origGraph, node_1);
    GraphEditor::addNode(origGraph, node_2);

    // TODO [SW-20488] Note that this is only done for the side effect
    // of creating an exec order since AllLivenessAnalysis uses the const
    // variation of getExeSortedNodes.
    UNUSED(origGraph.getExeSortedNodes());
    pLivenessAnalysis ls(new AllLivenessAnalysis(&origGraph));

    // wasRealTensorEncounteredBeforeNode
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_0), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_1), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_2), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_3), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_5), false);

    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_0), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_1), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_2), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_3), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_5), false);

    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_0), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_1), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_2), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_3), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_5), false);

    // isRealTensorAliveAfterNode
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_2), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_5), false);

    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_2), true);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_5), false);

    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_2), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_5), false);
}

TEST_F(LaTest, persistent_input_reuse_ends_lifetime_at_final_consumer)
{
    TSize sizes[] = {4};
    TensorPtr input(new Tensor(1, sizes, syn_type_float));
    TensorPtr intermediate0(new Tensor(1, sizes, syn_type_float));
    TensorPtr intermediate1(new Tensor(1, sizes, syn_type_float));
    TensorPtr output(new Tensor(1, sizes, syn_type_float));

    input->setMemoryDescriptor(synMemoryDescriptor(true));
    input->setTensorAsReusable(true);

    GaudiGraph graph;
    auto node0 = NodeFactory::createNode({input}, {intermediate0}, nullptr, NodeFactory::memcpyNodeTypeName, "reuse_0");
    auto node1 = NodeFactory::createNode(
        {intermediate0}, {intermediate1}, nullptr, NodeFactory::memcpyNodeTypeName, "reuse_1");
    auto node2 =
        NodeFactory::createNode({intermediate1}, {output}, nullptr, NodeFactory::memcpyNodeTypeName, "reuse_2");
    GraphEditor::addNode(graph, node0);
    GraphEditor::addNode(graph, node1);
    GraphEditor::addNode(graph, node2);

    UNUSED(graph.getExeSortedNodes());
    AllLivenessAnalysis reusableAnalysis(&graph);
    const Lifetime reusableLifetime = reusableAnalysis.getTensorLifeTime(input);
    EXPECT_EQ(graph.getMinExecutionOrderedIndex(), reusableLifetime.m_start);
    EXPECT_EQ(node0->getExecutionOrderedIndex(), reusableLifetime.m_end);
    EXPECT_LT(reusableLifetime.m_end, graph.getMaxExecutionOrderedIndex());

    input->setTensorAsReusable(false);
    AllLivenessAnalysis ordinaryAnalysis(&graph);
    const Lifetime ordinaryLifetime = ordinaryAnalysis.getTensorLifeTime(input);
    EXPECT_EQ(graph.getMinExecutionOrderedIndex(), ordinaryLifetime.m_start);
    EXPECT_EQ(graph.getMaxExecutionOrderedIndex(), ordinaryLifetime.m_end);
}

TEST_F(LaTest, minimal_latency_output_prioritizes_an_independent_ready_node)
{
    TSize sizes[] = {4};
    TensorPtr slowInput(new Tensor(1, sizes, syn_type_float));
    TensorPtr slowOutput(new Tensor(1, sizes, syn_type_float));
    TensorPtr fastInput(new Tensor(1, sizes, syn_type_float));
    TensorPtr fastOutput(new Tensor(1, sizes, syn_type_float));
    fastOutput->setMinimalLatency(true);

    GaudiGraph graph;
    auto slow =
        NodeFactory::createNode({slowInput}, {slowOutput}, nullptr, NodeFactory::memcpyNodeTypeName, "slow_ready");
    auto fast =
        NodeFactory::createNode({fastInput}, {fastOutput}, nullptr, NodeFactory::memcpyNodeTypeName, "fast_ready");
    GraphEditor::addNode(graph, slow);
    GraphEditor::addNode(graph, fast);

    const NodeVector& schedule = graph.getExeSortedNodes();
    ASSERT_EQ(2u, schedule.size());
    EXPECT_EQ(fast, schedule.front());
}
