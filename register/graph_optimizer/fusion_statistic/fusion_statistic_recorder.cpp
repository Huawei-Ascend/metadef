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

#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "graph/debug/ge_log.h"

namespace fe {

FusionStatisticRecorder::FusionStatisticRecorder(){};

FusionStatisticRecorder::~FusionStatisticRecorder(){};

FusionStatisticRecorder &FusionStatisticRecorder::Instance() {
  static FusionStatisticRecorder fusionStatisticRecoder;
  return fusionStatisticRecoder;
}

void FusionStatisticRecorder::UpdateGraphFusionMatchTimes(FusionInfo &fusionInfo) {
  std::lock_guard<std::recursive_mutex> lockGuard(mutex_);
  if (fusionInfo.GetMatchTimes() != 0) {
    std::string sessionAndGraphId = std::to_string(fusionInfo.GetSessionId()) + "_" + fusionInfo.GetGraphId();
    graphFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].AddMatchTimes(fusionInfo.GetMatchTimes());
    GELOGD("session %d graph %s pass %s matchTimes value: %d", fusionInfo.GetSessionId(),
           fusionInfo.GetGraphId().c_str(), fusionInfo.GetPassName().c_str(),
           graphFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].GetMatchTimes());
  }
}

void FusionStatisticRecorder::UpdateGraphFusionEffectTimes(FusionInfo &fusionInfo) {
  std::lock_guard<std::recursive_mutex> lockGuard(mutex_);
  if (fusionInfo.GetEffectTimes() != 0) {
    std::string sessionAndGraphId = std::to_string(fusionInfo.GetSessionId()) + "_" + fusionInfo.GetGraphId();
    graphFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].AddEffectTimes(fusionInfo.GetEffectTimes());
    GELOGD("session %d graph %s pass %s effectTimes value: %d", fusionInfo.GetSessionId(),
           fusionInfo.GetGraphId().c_str(), fusionInfo.GetPassName().c_str(),
           graphFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::UpdateBufferFusionMatchTimes(FusionInfo &fusionInfo) {
  std::lock_guard<std::recursive_mutex> lockGuard(mutex_);
  if (fusionInfo.GetMatchTimes() != 0) {
    std::string sessionAndGraphId = std::to_string(fusionInfo.GetSessionId()) + "_" + fusionInfo.GetGraphId();
    bufferFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].AddMatchTimes(fusionInfo.GetMatchTimes());
    GELOGD("ub session %d graph %s pass %s matchTimes value: %d", fusionInfo.GetSessionId(),
           fusionInfo.GetGraphId().c_str(), fusionInfo.GetPassName().c_str(),
           bufferFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].GetMatchTimes());
  }
}

void FusionStatisticRecorder::UpdateBufferFusionEffectTimes(FusionInfo &fusionInfo) {
  std::lock_guard<std::recursive_mutex> lockGuard(mutex_);
  if (fusionInfo.GetEffectTimes() != 0) {
    std::string sessionAndGraphId = std::to_string(fusionInfo.GetSessionId()) + "_" + fusionInfo.GetGraphId();
    bufferFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].AddEffectTimes(fusionInfo.GetEffectTimes());
    GELOGD("ub session %d graph %s pass %s effectTimes value: %d", fusionInfo.GetSessionId(),
           fusionInfo.GetGraphId().c_str(), fusionInfo.GetPassName().c_str(),
           bufferFusionInfoMap_[sessionAndGraphId][fusionInfo.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::GetAndClearFusionInfo(const std::string &sessionGraphId,
                                                    std::map<std::string, FusionInfo> &graphFusionInfoMap,
                                                    std::map<std::string, FusionInfo> &bufferFusionInfoMap) {
  std::lock_guard<std::recursive_mutex> lockGuard(mutex_);
  GELOGD("start to get graph map size %d", graphFusionInfoMap_.size());
  GELOGD("start to get ub graph map size %d", bufferFusionInfoMap_.size());
  GetFusionInfo(sessionGraphId, graphFusionInfoMap, bufferFusionInfoMap);
  ClearFusionInfo(sessionGraphId);
}

void FusionStatisticRecorder::GetFusionInfo(const std::string &sessionGraphId,
                                            std::map<std::string, FusionInfo> &graphFusionInfoMap,
                                            std::map<std::string, FusionInfo> &bufferFusionInfoMap) {
  if (graphFusionInfoMap_.find(sessionGraphId) != graphFusionInfoMap_.end()) {
    graphFusionInfoMap = graphFusionInfoMap_[sessionGraphId];
  }
  if (bufferFusionInfoMap_.find(sessionGraphId) != bufferFusionInfoMap_.end()) {
    bufferFusionInfoMap = bufferFusionInfoMap_[sessionGraphId];
  }
}

void FusionStatisticRecorder::ClearFusionInfo(std::string sessionGraphId) {
  if (graphFusionInfoMap_.find(sessionGraphId) != graphFusionInfoMap_.end()) {
    graphFusionInfoMap_.erase(sessionGraphId);
  }
  if (bufferFusionInfoMap_.find(sessionGraphId) != bufferFusionInfoMap_.end()) {
    bufferFusionInfoMap_.erase(sessionGraphId);
  }
}

FusionInfo::FusionInfo(uint64_t sessionId, std::string graphId, std::string passName, int32_t matchTimes,
                       int32_t effectTimes)
    : sessionId_(sessionId),
      graphId_(std::move(graphId)),
      passName_(std::move(passName)),
      matchTimes_(matchTimes),
      effectTimes_(effectTimes) {}

FusionInfo::~FusionInfo() {}

void FusionInfo::AddMatchTimes(int32_t matchTimes) { this->matchTimes_ += matchTimes; }

void FusionInfo::AddEffectTimes(int32_t effectTimes) { this->effectTimes_ += effectTimes; }

int32_t FusionInfo::GetMatchTimes() { return matchTimes_; }

int32_t FusionInfo::GetEffectTimes() { return effectTimes_; }

std::string FusionInfo::GetGraphId() { return graphId_; }

std::string FusionInfo::GetPassName() { return passName_; }

uint64_t FusionInfo::GetSessionId() { return sessionId_; }

void FusionInfo::SetMatchTimes(int32_t matchTimes) { this->matchTimes_ = matchTimes; }

void FusionInfo::SetEffectTimes(int32_t effectTimes) { this->effectTimes_ = effectTimes; }
}
