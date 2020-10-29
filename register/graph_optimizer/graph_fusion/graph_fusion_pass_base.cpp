/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include <memory>
#include <string>
#include <vector>
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"

namespace fe {
GraphFusionPassBase::GraphFusionPassBase() {
  patternFusionBasePassImplPtr_ = std::make_shared<PatternFusionBasePassImpl>();
}

GraphFusionPassBase::~GraphFusionPassBase() {}

/**
 * @ingroup fe
 * @brief execute pass
 */
Status GraphFusionPassBase::Run(ge::ComputeGraph &graph) {
  Mappings mappings;
  bool is_patterns_ok = true;
  // build Pattern
  vector<FusionPattern *> patterns;
  patternFusionBasePassImplPtr_->GetPatterns(patterns);
  if (patterns.empty()) {
    patterns = DefinePatterns();
    for (FusionPattern *pattern : patterns) {
      if (pattern != nullptr) {
        bool ok = pattern->Build();
        if (!ok) {
          GELOGW("this pattern: %s build not success.", pattern->GetName().c_str());
        }
        pattern->Dump();
        is_patterns_ok = is_patterns_ok && ok;
      }
    }

    patternFusionBasePassImplPtr_->SetPatterns(patterns);
  }
  if (!is_patterns_ok) {
    GELOGE(FAILED, "Patterns invalid.");
    return FAILED;
  }

  NodeMapInfoPtr nodeMapInfo = nullptr;
  if (GraphPassUtil::GetOpTypeMapToGraph(nodeMapInfo, graph) == SUCCESS) {
    nodeMapInfo->runCount++;
  }
  // do matching and fusion for each pattern
  bool finalChanged = false;
  for (const FusionPattern *pattern : patterns) {
    if (pattern != nullptr) {
      bool changed = false;
      Status ret = RunOnePattern(graph, *pattern, changed);
      if (ret != SUCCESS) {
        GELOGW("run pattern %s not success, graph is not changed by it.", pattern->GetName().c_str());
        return ret;
      }
      finalChanged = finalChanged || changed;
    }
  }
  return finalChanged ? SUCCESS : NOT_CHANGED;
}

/**
 * @ingroup fe
 * @brief do matching and fusion in graph based on the pattern
 */
Status GraphFusionPassBase::RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed) {
  changed = false;
  Mappings mappings;
  int32_t effectTimes = 0;
  uint32_t graphId = graph.GetGraphID();
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graphId), GetName(), static_cast<int32_t>(mappings.size()),
                        effectTimes);
  // match all patterns in graph, and save them to mappings
  if (!MatchAll(graph, pattern, mappings)) {
    GELOGD("GraphFusionPass[%s]: pattern=%s, matchedTimes=%zu, effectedTimes=%d.", GetName().c_str(),
           pattern.GetName().c_str(), mappings.size(), effectTimes);
    return SUCCESS;
  }

  GELOGD(
      "This graph has been matched with pattern[%s]."
      "The mappings are as follows.",
      pattern.GetName().c_str());

  // print the results of matching
  patternFusionBasePassImplPtr_->DumpMappings(pattern, mappings);
  NodeMapInfoPtr nodeMapInfo = nullptr;
  // get nodes by type from node
  (void)GraphPassUtil::GetOpTypeMapToGraph(nodeMapInfo, graph);
  // do fusion for each mapping
  for (Mapping &mapping : mappings) {
    vector<ge::NodePtr> fusNodes;
    ge::NodePtr firstNode = nullptr;
    for (auto &item : mapping) {
      if (!item.second.empty()) {
        firstNode = item.second[0];
        break;
      }
    }

    Status status = Fusion(graph, mapping, fusNodes);
    if (status != SUCCESS && status != NOT_CHANGED) {
      GELOGE(status, "Fail to fuse the graph with pattern[%s].", pattern.GetName().c_str());
      return status;
    }

    if (status == SUCCESS) {
      effectTimes++;
      if (!fusNodes.empty()) {
        // add fusednode to node map info
        for (ge::NodePtr &node : fusNodes) {
          GraphPassUtil::AddNodeFromOpTypeMap(nodeMapInfo, node);
        }
      }
    }
    changed = changed || status == SUCCESS;
  }

  // get match times and effect times
  FusionStatisticRecorder &fusionStatisticInst = FusionStatisticRecorder::Instance();
  fusionInfo.SetMatchTimes(static_cast<int32_t>(mappings.size()));
  fusionInfo.SetEffectTimes(effectTimes);
  fusionStatisticInst.UpdateGraphFusionMatchTimes(fusionInfo);
  fusionStatisticInst.UpdateGraphFusionEffectTimes(fusionInfo);
  GELOGD("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matchedTimes=%d, effectedTimes=%d.", graphId, GetName().c_str(),
         pattern.GetName().c_str(), static_cast<int32_t>(mappings.size()), effectTimes);
  return SUCCESS;
}

/**
 * @ingroup fe
 * @brief match all nodes in graph according to pattern
 */
// match nodes in graph according to pattern, the algorithm is shown as
// following:
// 1. get output node from pattern
// 2. Search for candidate nodes in Graph (network Graph generated after
//    parsing) according to Op Type and
// (optional), and add the candidate node to the list of candidates
// 3. For each Node in the candidate list, check whether the type and the number
//    of precursors are consistent with the description of corresponding Op
//    in pattern. If they are consistent, add the precursor Node to the
//    candidate list, and add "PatternOp-GraphNode" to the mapping; otherwise,
//    return an empty mapping
// 4. repeat step 3 until all the Ops in pattern are matched
// 5. if all the Ops in pattern are matched successfully, return the mapping of
//    PatternOp and GraphNode
bool GraphFusionPassBase::MatchAll(ge::ComputeGraph &graph, const FusionPattern &pattern, Mappings &mappings) {
  vector<ge::NodePtr> matchedOutputNodes;

  // find all the output nodes of pattern in the graph based on Op type
  std::shared_ptr<FusionPattern::OpDesc> outputOpDesc = pattern.GetOutput();
  if (outputOpDesc == nullptr) {
    return false;
  }

  if (!patternFusionBasePassImplPtr_->GetMatchOutputNodes(graph, pattern, matchedOutputNodes)) {
    return false;
  }

  // begin matching from every output node
  for (ge::NodePtr &outputNode : matchedOutputNodes) {
    Mapping mapping;
    if (patternFusionBasePassImplPtr_->MatchFromOutput(outputNode, outputOpDesc, mapping)) {
      mappings.push_back(mapping);
    }
  }
  // if matching is successful, return true; otherwise false
  return !mappings.empty();
}

/**
 * @ingroup fe
 * @brief get an op from mapping according to ID
 */
ge::NodePtr GraphFusionPassBase::GetNodeFromMapping(const string &id, const Mapping &mapping) {
  for (auto &item : mapping) {
    std::shared_ptr<OpDesc> opDesc = item.first;
    if (opDesc != nullptr && opDesc->id == id) {
      if (item.second.empty()) {
        return nullptr;
      } else {
        return item.second[0];
      }
    }
  }
  return nullptr;
}

}  // namespace fe
