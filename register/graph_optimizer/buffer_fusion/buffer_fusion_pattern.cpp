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

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include <string>
#include <vector>
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

using std::map;
using std::string;
using std::vector;

namespace fe {
inline bool IsAddOverflow(int64_t a, int64_t b) {
  return ((b > 0) && (a > ((int64_t)INT64_MAX - b))) || ((b < 0) && (a < ((int64_t)INT64_MIN - b)));
}

BufferFusionPattern::BufferFusionPattern(string name, int64_t maxCount)
    : name_(name), opMaxCount_(maxCount), errorCount_(0) {}

BufferFusionPattern::~BufferFusionPattern() {
  for (auto op : ops_) {
    delete (op);
  }
}

/*
 * @brief:  add op desc info
 * @param [in] descName: node desc name
 * @param [in] types: node desc type
 * @param [in] repeateMin: the min count for fusion match,
 *                         patter match failed if real count lower than the
 * value
 * @param [in] repeateMax: the max count for fusion match,
 *                         the op will be ignored if current match count equla
 * with the value
 * @return BufferFusionPattern: pattern object
 */

BufferFusionPattern &BufferFusionPattern::AddOpDesc(const std::string &descName, const std::vector<std::string> &types,
                                                    int64_t repeateMin, int64_t repeateMax, int64_t groupId) {
  if (descName.empty()) {
    GELOGW("Desc_name cannot be empty.");
    errorCount_++;
    return *this;
  }

  if (repeateMin > repeateMax) {
    GELOGW(
        "Repeat_min can not lager than repeateMax, desc name is [%s], min is "
        "[%ld], max is [%ld]",
        descName.c_str(), repeateMin, repeateMax);
    errorCount_++;
    return *this;
  }

  if (GetOpDesc(descName) != nullptr) {
    GELOGW("Desc_name repeated. (descName:%s)", descName.c_str());
    errorCount_++;
    return *this;
  }

  BufferFusionOpDesc *op = new (std::nothrow) BufferFusionOpDesc();
  if (op == nullptr) {
    GELOGW("New an object failed.");
    errorCount_++;
    return *this;
  }

  op->descName = descName;
  op->types = types;
  op->repeateMin = repeateMin;
  op->repeateMax = repeateMax;
  op->repeateCurr = 0;
  op->groupId = groupId;
  op->matchStatus = false;
  op->outBranchType = TBE_OUTPUT_BRANCH_DEFAULT;
  op->ignoreInputNum = false;
  op->ignoreOutputNum = false;
  if (repeateMax > repeateMin) {
    for (int64_t i = repeateMin; i < repeateMax; i++) {
      op->multiOutputSkipStatus.insert(std::pair<int64_t, SkipStatus>(i, SkipStatus::DISABLED));
    }
  }
  ops_.push_back(op);
  opMap_[descName] = op;

  op->outputs.clear();
  return *this;
}

/*
 * @brief:  set output desc info
 * @param [in] descName: node desc name
 * @param [in] output_ids: output desc
 * @param [in] relation:   output desc relation (1: serial, 2:parallel)
 * @return BufferFusionPattern: pattern object
 */
BufferFusionPattern &BufferFusionPattern::SetOutputs(const string &descName, const std::vector<string> &output_ids,
                                                     int64_t relation, bool ignoreInputNum, bool ignoreOutputNum) {
  if (descName.empty()) {
    GELOGW("Desc_name cannot be empty.");
    errorCount_++;
    return *this;
  }

  BufferFusionOpDesc *opDesc = GetOpDesc(descName);
  if (opDesc == nullptr) {
    GELOGW("Desc_name not exist. (descName:%s)", descName.c_str());
    errorCount_++;
    return *this;
  }

  opDesc->ignoreInputNum = ignoreInputNum;
  opDesc->ignoreOutputNum = ignoreOutputNum;
  if (opDesc->outBranchType == TBE_OUTPUT_BRANCH_DEFAULT) {
    opDesc->outBranchType = relation;
  }

  UpdateSkipStatus(opDesc);

  // support one multi output for one optype
  for (const string &output_id : output_ids) {
    BufferFusionOpDesc *output_op_desc = GetOpDesc(output_id);
    if (output_op_desc == nullptr) {
      GELOGW("Desc_name not exist. (descName:%s)", descName.c_str());
      if (IsAddOverflow(errorCount_, 1) != SUCCESS) {
        GELOGW("errorCount_++ overflow. (descName:%s)", descName.c_str());
        return *this;
      }
      errorCount_++;
      return *this;
    }
    if (opDesc == output_op_desc) {
      continue;
    }

    opDesc->outputs.push_back(output_op_desc);
    output_op_desc->inputs.push_back(opDesc);

    if (opDesc->outBranchType != relation) {
      GELOGW("Failed to set outputs relation: curr is [%ld], new is [%ld].", opDesc->outBranchType, relation);
      return *this;
    }
  }
  return *this;
}

/*
 * @brief:  get output desc info
 * @param [in]  opDesc: current desc
 * @param [out] outputs: candidate output desc set
 * @return bool: get output desc ok or not
 */
bool BufferFusionPattern::GetOutputs(BufferFusionOpDesc *opDesc, std::vector<BufferFusionOpDesc *> &outputs,
                                     bool ignoreRepeat) {
  if (opDesc == nullptr) {
    GELOGW("failed to get outputs: opDesc is null.");
    return false;
  }
  string desc_n = opDesc->descName;

  // add curr desc can be reused while repeateCurr < repeateMax
  if (!ignoreRepeat && opDesc->repeateCurr < opDesc->repeateMax) {
    outputs.push_back(opDesc);
  }

  // check candidate desc
  for (auto desc : opDesc->outputs) {
    if (desc == nullptr) {
      GELOGD("desc[%s] has null output desc.", desc_n.c_str());
      continue;
    }
    // add out desc
    outputs.push_back(desc);

    // add sub outdescs while repeateMin == 0
    if (desc->repeateMin == 0) {
      std::vector<BufferFusionOpDesc *> sub_output;
      if (GetOutputs(desc, sub_output, true)) {
        for (auto sub_desc : sub_output) {
          outputs.push_back(sub_desc);
        }
      }
    }
  }

  return true;
}

/*
 * @brief: set fusion pattern head
 * @param [in] head_ids: node list
 * @return bool: set head desc ok or not
 */
BufferFusionPattern &BufferFusionPattern::SetHead(const std::vector<string> &head_ids) {
  if (head_ids.empty()) {
    GELOGW("input vector is empty.");
    errorCount_++;
    return *this;
  }
  for (const string &head_id : head_ids) {
    BufferFusionOpDesc *head_op_desc = GetOpDesc(head_id);
    if (head_op_desc == nullptr) {
      GELOGW("descName not exist. (descName:%s)", head_id.c_str());
      if (IsAddOverflow(errorCount_, 1) != SUCCESS) {
        GELOGW("errorCount_++ overflow. (descName:%s)", head_id.c_str());
        return *this;
      }
      errorCount_++;
      return *this;
    }
    // Head desc repeat number can not excceed 1
    // if must be excceed 1, it can be realized by several descs
    if (head_op_desc->repeateMax > 1) {
      GELOGW(
          "Head desc repeat number can not excceed 1, head desc name is [%s], "
          "actual repeateMax is [%ld]",
          head_id.c_str(), head_op_desc->repeateMax);
      if (IsAddOverflow(errorCount_, 1) != SUCCESS) {
        GELOGW("errorCount_++ overflow. (descName:%s)", head_id.c_str());
        return *this;
      }
      errorCount_++;
      return *this;
    }
    head_.push_back(head_op_desc);
  }

  // check head desc repeat min total value, it can not excceed 1
  int64_t desc_total_min = 0;
  for (auto desc : head_) {
    if (IsAddOverflow(desc_total_min, desc->repeateMin) != SUCCESS) {
      GELOGW("desc_total_min+repeateMin overflow.");
      return *this;
    }
    desc_total_min += desc->repeateMin;
  }

  if (desc_total_min > 1) {
    GELOGW(
        "head desc repeat min total value can not be larger than 1, current is "
        "[%ld]",
        desc_total_min);
    errorCount_++;
    return *this;
  }
  return *this;
}

void BufferFusionPattern::UpdateSkipStatus(BufferFusionOpDesc *opDesc) {
  if (opDesc->outBranchType == TBE_OUTPUT_BRANCH_MULTI) {
    for (auto &inputdesc : opDesc->inputs) {
      if (inputdesc->types.size() != opDesc->types.size()) {
        continue;
      }
      bool isSameType = true;
      for (uint32_t i = 0; i < inputdesc->types.size(); i++) {
        if (inputdesc->types[i] != opDesc->types[i]) {
          isSameType = false;
          break;
        }
      }
      if (isSameType && inputdesc->ignoreOutputNum == true) {
        for (uint32_t i = inputdesc->repeateMin; i < inputdesc->repeateMax; i++) {
          inputdesc->multiOutputSkipStatus[i] = SkipStatus::AVAILABLE;
        }
      }
    }
  }
}

/*
 * @brief: get description ptr by name
 * @param [in] descName: fusion pattern desc name
 * @return BufferFusionOpDesc*: description ptr
 */
BufferFusionOpDesc *BufferFusionPattern::GetOpDesc(const string &descName) {
  auto it = opMap_.find(descName);
  if (it != opMap_.end()) return it->second;

  return nullptr;
}

std::vector<BufferFusionOpDesc *> BufferFusionPattern::GetHead() { return head_; }

std::string BufferFusionPattern::GetName() { return name_; }
int64_t BufferFusionPattern::GetOpMaxCount() { return opMaxCount_; }
int64_t BufferFusionPattern::GetErrorCnt() { return errorCount_; }

std::vector<BufferFusionOpDesc *> BufferFusionPattern::GetOpDescs() { return ops_; }
}  // namespace fe
