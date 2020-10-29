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

#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
PatternFusionBasePassImpl::PatternFusionBasePassImpl() {}

PatternFusionBasePassImpl::~PatternFusionBasePassImpl() {
  for (auto pattern : patterns_) {
    if (pattern != nullptr) {
      delete pattern;
      pattern = nullptr;
    }
  }
}

void PatternFusionBasePassImpl::GetPatterns(vector<FusionPattern *> &patterns) { patterns = patterns_; }

void PatternFusionBasePassImpl::SetPatterns(vector<FusionPattern *> &patterns) { patterns_ = patterns; }

void PatternFusionBasePassImpl::SetOpsKernelInfoStore(OpsKernelInfoStorePtr opsKernelInfoStorePtr) {
  opsKernelInfoStorePtr_ = opsKernelInfoStorePtr;
}

bool PatternFusionBasePassImpl::CheckOpSupported(const ge::OpDescPtr &opDescPtr) {
  bool result = false;

  std::string unSupportedReason;

  if (opsKernelInfoStorePtr_ == nullptr) {
    unSupportedReason = "opsKernelInfoStorePtr in PatternFusionBasePass is nullptr.";
    return false;
  }

  result = opsKernelInfoStorePtr_->CheckSupported(opDescPtr, unSupportedReason);
  return result;
}

bool PatternFusionBasePassImpl::IsNodesExist(ge::NodePtr currentNode, std::vector<ge::NodePtr> &nodes) {
  return find(nodes.begin(), nodes.end(), currentNode) != nodes.end();
}

bool PatternFusionBasePassImpl::IsMatched(std::shared_ptr<OpDesc> opDesc, const ge::NodePtr node,
                                          const Mapping &mapping) {
  if (opDesc == nullptr || node == nullptr) {
    GELOGD("opDesc or node could not be null");
    return false;
  }

  const auto iter = mapping.find(opDesc);

  // check opDesc does not exist in mapping
  return iter != mapping.end() && (find(iter->second.begin(), iter->second.end(), node) != iter->second.end());
}

void PatternFusionBasePassImpl::DumpMappings(const FusionPattern &pattern, const Mappings &mappings) {
  std::ostringstream oss;
  oss << std::endl << "Mappings of pattern ";
  oss << pattern.GetName() << ":" << std::endl;
  for (size_t i = 0; i < mappings.size(); i++) {
    const Mapping &mapping = mappings[i];
    oss << " Mapping " << (i + 1) << "/" << mappings.size() << ":" << std::endl;
    for (const auto &item : mapping) {
      std::shared_ptr<OpDesc> opDesc = item.first;
      const ge::NodePtr node = item.second[0];
      if (opDesc != nullptr && node != nullptr) {
        oss << "    " << opDesc->id << " -> " << node->GetName() << std::endl;
      }
    }
  }
  GELOGD("%s", oss.str().c_str());
}

bool PatternFusionBasePassImpl::IsOpTypeExist(const string &type, const vector<string> &types) {
  return find(types.begin(), types.end(), type) != types.end();
}

bool PatternFusionBasePassImpl::MatchFromOutput(ge::NodePtr outputNode, std::shared_ptr<OpDesc> outputOpDesc,
                                                Mapping &mapping) {
  if (outputNode == nullptr) {
    GELOGW("outputNode is null, pattern matching failed");
    return false;
  }

  if (outputOpDesc == nullptr) {
    GELOGW("outputOpDesc is null, pattern matching failed");
    return false;
  }

  vector<ge::NodePtr> candidateNodes = {outputNode};
  vector<std::shared_ptr<OpDesc>> candidateOpDescs = {outputOpDesc};

  // store the nodes matched
  mapping[outputOpDesc].push_back(outputNode);

  // match candidate node one by one
  while (!candidateNodes.empty() && !candidateOpDescs.empty()) {
    // get the first candidate node
    bool result = MatchFromOutput(candidateNodes, candidateOpDescs, mapping);
    if (!result) {
      return false;
    }

    // current op is matched successfully, thus remove it from candidate list
    candidateNodes.erase(candidateNodes.begin());
    candidateOpDescs.erase(candidateOpDescs.begin());

    // the sizes of candidateNodes and candidateOpDescs should always keep the same
    if (candidateNodes.size() != candidateOpDescs.size()) {
      GELOGW("candidateNodes size does not equal to candidateOpDescs size, pattern matching failed.");
      return false;
    }
  }

  // if candidateNodes(or candidateOpDescs) is empty, the matching is done
  // successfully
  return candidateOpDescs.empty();
}

bool PatternFusionBasePassImpl::MatchFromOutput(vector<ge::NodePtr> &candidateNodes,
                                                vector<std::shared_ptr<OpDesc>> &candidateOpDescs, Mapping &mapping) {
  if (candidateNodes.empty() || candidateOpDescs.empty()) {
    GELOGW("candidateNodes or candidateOpDescs is empty, pattern matching failed.");
    return false;
  }
  ge::NodePtr node = candidateNodes.front();
  std::shared_ptr<OpDesc> opDesc = candidateOpDescs.front();
  string opId = opDesc->id;
  // add the input nodes into candidate list
  const vector<std::shared_ptr<OpDesc>> *inputs_desc = FusionPattern::GetInputs(opDesc);
  if (inputs_desc == nullptr) {
    GELOGW("Op[%s]: the inputs_desc is null, pattern matching failed.", opId.c_str());
    return false;
  }

  if (inputs_desc->empty()) {
    return true;
  }

  if (node->GetInDataNodes().empty()) {
    GELOGW("Op[%s]: in data node or inputs_desc is empty, pattern matching failed.", opId.c_str());
    return false;
  }

  // set flag for edge using
  const std::unique_ptr<bool[]> usage_flags(new (std::nothrow) bool[inputs_desc->size()]{});

  // order the input edges, and the order should also be the rule of pattern
  // setting
  std::vector<ge::InDataAnchorPtr> in_anchors;
  GetInDataAnchors(node, in_anchors);
  if (in_anchors.empty()) {
    GELOGW("Op[%s]: in data anchors are empty, pattern matching failed.", opId.c_str());
    return false;
  }

  std::sort(in_anchors.begin(), in_anchors.end(),
            [](ge::InDataAnchorPtr a, ge::InDataAnchorPtr b) { return a->GetIdx() < b->GetIdx(); });

  for (auto in_anchor : in_anchors) {
    ge::NodePtr input_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    for (uint32_t j = 0; j < inputs_desc->size(); j++) {
      std::shared_ptr<OpDesc> input_desc = inputs_desc->at(j);
      if (input_desc == nullptr) {
        GELOGW("Op[%s]: input_desc is null, pattern matching failed.", opId.c_str());
        return false;
      }

      bool condi =
          (IsOpTypeExist(ge::NodeUtils::GetNodeType(*input_node), input_desc->types) || input_desc->types.empty()) &&
          (!usage_flags[j] || input_desc->repeatable);
      if (!condi) {
        continue;
      }
      // some nodes might be the input of multiple nodes, we use
      // IsMatched() to avoid repeat
      if (!IsMatched(input_desc, input_node, mapping)) {
        candidateNodes.push_back(input_node);
        candidateOpDescs.push_back(input_desc);
        // store the matched node
        mapping[input_desc].push_back(input_node);
      }
      usage_flags[j] = true;
      break;
    }
  }

  // return false if not all edges are matched
  if (!MatchAllEdges(inputs_desc->size(), usage_flags)) {
    GELOGW("Op[%s]: not all inputs are matched, pattern matching failed.", opId.c_str());
    return false;
  }
  return true;
}

bool PatternFusionBasePassImpl::MatchAllEdges(const size_t &input_size, const std::unique_ptr<bool[]> &usage_flags) {
  for (size_t i = 0; i != input_size; i++) {
    if (!usage_flags[i]) {
      return false;
    }
  }
  return true;
}

void PatternFusionBasePassImpl::GetInDataAnchors(const ge::NodePtr &node,
                                                 std::vector<ge::InDataAnchorPtr> &in_anchor_vec) {
  for (auto in_anchor : node->GetAllInDataAnchors()) {
    if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr ||
        in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    in_anchor_vec.push_back(in_anchor);
  }
}

bool PatternFusionBasePassImpl::GetMatchOutputNodes(ge::ComputeGraph &graph, const FusionPattern &pattern,
                                                    vector<ge::NodePtr> &matchedOutputNodes) {
  std::shared_ptr<FusionPattern::OpDesc> outputOpDesc = pattern.GetOutput();
  if (outputOpDesc == nullptr) {
    GELOGW("outputOpDesc is null, pattern matching failed");
    return false;
  }

  NodeMapInfoPtr nodeMapInfo = nullptr;
  // get nodes by type from node
  if (GraphPassUtil::GetOpTypeMapToGraph(nodeMapInfo, graph) == SUCCESS) {
    for (auto &OutOpType : outputOpDesc->types) {
      auto iter = nodeMapInfo->nodeTypeMap->find(OutOpType);
      if (iter != nodeMapInfo->nodeTypeMap->end()) {
        for (auto &nodePtr : iter->second) {
          if (nodePtr->GetInDataNodes().empty() && nodePtr->GetOutAllNodes().empty()) {
            continue;
          }
          if (ge::NodeUtils::GetNodeType(*nodePtr) == OutOpType) {
            matchedOutputNodes.push_back(nodePtr);
          }
        }
      }
    }
  } else {  // for each graph to find type
    for (ge::NodePtr &n : graph.GetDirectNode()) {
      if (n == nullptr) {
        GELOGW("node from graph is null, pattern matching failed");
        return false;
      }

      if (IsOpTypeExist(ge::NodeUtils::GetNodeType(*n), outputOpDesc->types)) {
        matchedOutputNodes.push_back(n);
      }
    }
  }

  if (matchedOutputNodes.empty()) {
    return false;
  }
  return true;
}
}
