#include "graph.hpp"
#include "../utils/logger.hpp"
#include <set>
#include <queue>

namespace onnx_runner {

void Graph::printSummary() const {
    LOG_INFO("=== Graph Summary ===");
    LOG_INFO("Nodes: ", nodes_.size());
    LOG_INFO("Initializers: ", initializers_.size());
    LOG_INFO("Inputs: ", inputs_.size());
    LOG_INFO("Outputs: ", outputs_.size());

    LOG_INFO("\n=== Graph Inputs ===");
    for (const auto& input : inputs_) {
        LOG_INFO("  - ", input);
    }

    LOG_INFO("\n=== Graph Outputs ===");
    for (const auto& output : outputs_) {
        LOG_INFO("  - ", output);
    }

    LOG_INFO("\n=== Initializers ===");
    for (const auto& [name, tensor] : initializers_) {
        LOG_INFO("  - ", name, ": ", tensor->shapeStr());
    }

    LOG_INFO("\n=== Nodes ===");
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        LOG_INFO("  [", i, "] ", opTypeToString(node->opType()), " (", node->name(), ")");
        LOG_INFO("      Inputs: ");
        for (const auto& input : node->inputs()) {
            LOG_INFO("        - ", input);
        }
        LOG_INFO("      Outputs: ");
        for (const auto& output : node->outputs()) {
            LOG_INFO("        - ", output);
        }
    }
}

std::vector<std::shared_ptr<Node>> Graph::topologicalSort() const {
    // Build dependency graph
    std::map<std::string, std::set<std::shared_ptr<Node>>> producers;  // output_name -> node that produces it
    std::map<std::shared_ptr<Node>, std::set<std::shared_ptr<Node>>> dependencies;  // node -> nodes it depends on
    std::map<std::shared_ptr<Node>, int> in_degree;

    // Initialize in_degree for all nodes
    for (const auto& node : nodes_) {
        in_degree[node] = 0;
    }

    // Build producer map
    for (const auto& node : nodes_) {
        for (const auto& output : node->outputs()) {
            producers[output].insert(node);
        }
    }

    // Build dependency map and in_degree
    for (const auto& node : nodes_) {
        for (const auto& input : node->inputs()) {
            // Skip initializers (constants don't create dependencies)
            if (isInitializer(input)) continue;

            // Find all nodes that produce this input
            if (producers.find(input) != producers.end()) {
                for (const auto& producer : producers[input]) {
                    if (dependencies[node].find(producer) == dependencies[node].end()) {
                        dependencies[node].insert(producer);
                        in_degree[node]++;
                    }
                }
            }
        }
    }

    // Kahn's algorithm for topological sort
    std::queue<std::shared_ptr<Node>> queue;
    std::vector<std::shared_ptr<Node>> result;

    // Add all nodes with no dependencies
    for (const auto& node : nodes_) {
        if (in_degree[node] == 0) {
            queue.push(node);
        }
    }

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();
        result.push_back(node);

        // For each node that depends on this one, decrease in_degree
        for (const auto& dependent : nodes_) {
            if (dependencies[dependent].find(node) != dependencies[dependent].end()) {
                in_degree[dependent]--;
                if (in_degree[dependent] == 0) {
                    queue.push(dependent);
                }
            }
        }
    }

    if (result.size() != nodes_.size()) {
        LOG_ERROR("Graph has cycles or unreachable nodes!");
        return nodes_;  // Return original order as fallback
    }

    return result;
}

} // namespace onnx_runner
