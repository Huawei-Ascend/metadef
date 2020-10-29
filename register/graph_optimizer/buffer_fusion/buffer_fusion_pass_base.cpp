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

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include <map>
#include <string>
#include <vector>

namespace fe {
BufferFusionPassBase::BufferFusionPassBase() {}

BufferFusionPassBase::~BufferFusionPassBase() {}

Status BufferFusionPassBase::GetFusionNodes(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusionNodes) {
  fusionNodes = GetMatchedNodes(mapping);
  return SUCCESS;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodes(const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    for (const auto &node : item.second) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodesByDescName(const std::string &descName,
                                                                         const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    const BufferFusionOpDesc *opDesc = item.first;
    if (opDesc != nullptr && opDesc->descName == descName) {
      for (const auto &node : item.second) {
        nodes.push_back(node);
      }
    }
  }
  return nodes;
}

ge::NodePtr BufferFusionPassBase::GetMatchedHeadNode(const std::vector<ge::NodePtr> &matchedNodes) {
  for (auto node : matchedNodes) {
    auto inputNodes = node->GetInDataNodes();
    bool findFlag = false;
    for (const auto &inNode : inputNodes) {
      // find the node from fuison sub graph
      if (std::find(matchedNodes.begin(), matchedNodes.end(), inNode) != matchedNodes.end()) {
        findFlag = true;
        break;
      }
    }
    if (findFlag == false) {
      return node;
    }
  }
  return nullptr;
}

}  // namespace fe
