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

#ifndef INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_

#include <map>
#include <string>

namespace ge {
class StatusFactory {
 public:
  static StatusFactory *Instance() {
    static StatusFactory instance;
    return &instance;
  }

  void RegisterErrorNo(uint32_t err, const std::string &desc) {
    // Avoid repeated addition
    if (err_desc_.find(err) != err_desc_.end()) {
      return;
    }
    err_desc_[err] = desc;
  }

  std::string GetErrDesc(uint32_t err) {
    auto iter_find = err_desc_.find(err);
    if (iter_find == err_desc_.end()) {
      return "";
    }
    return iter_find->second;
  }

 protected:
  StatusFactory() {}
  ~StatusFactory() {}

 private:
  std::map<uint32_t, std::string> err_desc_;
};

class ErrorNoRegisterar {
 public:
  ErrorNoRegisterar(uint32_t err, const std::string &desc) { StatusFactory::Instance()->RegisterErrorNo(err, desc); }
  ~ErrorNoRegisterar() {}
};

// Code compose(4 byte), runtime: 2 bit,  type: 2 bit,   level: 3 bit,  sysid: 8 bit, modid: 5 bit, value: 12 bit
#define GE_ERRORNO(runtime, type, level, sysid, modid, name, value, desc)                              \
  constexpr ge::Status name =                                                                          \
    ((0xFF & (static_cast<uint8_t>(runtime))) << 30) | ((0xFF & (static_cast<uint8_t>(type))) << 28) | \
    ((0xFF & (static_cast<uint8_t>(level))) << 25) | ((0xFF & (static_cast<uint8_t>(sysid))) << 17) |  \
    ((0xFF & (static_cast<uint8_t>(modid))) << 12) | (0x0FFF & (static_cast<uint16_t>(value)));        \
  const ErrorNoRegisterar g_##name##_errorno(name, desc);

#define GE_ERRORNO_EXTERNAL(name, errorcode, desc) \
  static const ge::Status name = errorcode;        \
  const ErrorNoRegisterar g_##name##_errorno(name, desc);

using Status = uint32_t;

// General error code
GE_ERRORNO(0, 0, 0, 0, 0, SUCCESS, 0, "success");
GE_ERRORNO(0b11, 0b11, 0b111, 0xFF, 0b11111, FAILED, 0xFFF, "failed"); /*lint !e401*/

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_PARAM_INVALID, 145000, "Parameter invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_NOT_INIT, 145001, "GE executor not initialized yet.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, 145002, "Model file path invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, 145003, "Model id invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_KEY_PATH_INVALID, 145004, "Model key path invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_NOT_SUPPORT_ENCRYPTION, 145005, "Model does not support encryption.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, 145006, "Data size of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, 145007, "Model addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, 145008, "Queue id of model invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, 145009, "The model loaded repeatedly.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_MODEL_PARTITION_NUM_INVALID, 145010, "Model partition num invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID, 145011, "Dynamic input addr invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID, 145012, "Dynamic input size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID, 145013, "Dynamic batch size invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_BATCH_EMPTY, 145014, "AIPP batch parameter empty.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_NOT_EXIST, 145015, "AIPP parameter not exist.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_AIPP_MODE_INVALID, 145016, "AIPP mode invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_TASK_TYPE_INVALID, 145017, "Task type invalid.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID, 145018, "Kernel type invalid.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_MEMORY_ALLOCATION, 245000, "Memory allocation error.");

GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_INTERNAL_ERROR, 545000, "Internal error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_LOAD_MODEL, 545001, "Load model error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED, 545002, "Failed to load model partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED, 545003, "Failed to load weight partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, 545004, "Failed to load task partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED, 545005, "Failed to load op kernel partition.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA, 545006, "Failed to release the model data.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_COMMAND_HANDLE, 545007, "Command handle error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_GET_TENSOR_INFO, 545008, "Get tensor info error.");
GE_ERRORNO_EXTERNAL(ACL_ERROR_GE_UNLOAD_MODEL, 545009, "Load model error.");

}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
