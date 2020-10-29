/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <chrono>
#include <random>
#include <cstring>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"

#include "register/op_tiling.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"


#define LOG_ENABLED(loglvl) CheckLogLevel(GE_MODULE_NAME, loglvl)

namespace optiling {

constexpr int UUID_LENGTH = 32;

const std::map<ge::DataType, std::string> DATATYPE_STRING_MAP {
  {ge::DT_FLOAT,     "float32"  },
  {ge::DT_FLOAT16,   "float16"  },
  {ge::DT_INT8,      "int8"     },
  {ge::DT_INT16,     "int16"    },
  {ge::DT_INT32,     "int32"    },
  {ge::DT_INT64,     "int64"    },
  {ge::DT_UINT8,     "uint8"    },
  {ge::DT_UINT16,    "uint16"   },
  {ge::DT_UINT32,    "uint32"   },
  {ge::DT_UINT64,    "uint64"   },
  {ge::DT_BOOL,      "bool"     },
  {ge::DT_DOUBLE,    "double"   },
  {ge::DT_DUAL,      "dual"     },
  {ge::DT_DUAL_SUB_INT8, "dual_sub_int8"},
  {ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8"}
};

std::string GenUuid()
{
    static std::random_device dev;
    static std::mt19937 rng(dev());
    const char *v = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::uniform_int_distribution<int> dist(0, std::strlen(v) - 1);
    char res[UUID_LENGTH + 1] = {0};
    for (size_t i = 0; i < sizeof(res) - 1; i++) {
        res[i] = v[dist(rng)];
    }

    return res;
}

std::string OpTilingInterf::OpTilingUuid = GenUuid();

std::map<std::string, OpTilingFunc> & OpTilingInterf::RegisteredOpInterf() {
    static std::map<std::string, OpTilingFunc> interf;
    return interf;
}


OpTilingInterf::OpTilingInterf(std::string op_type, OpTilingFunc func) {
    auto &interf = RegisteredOpInterf();
    interf.emplace(op_type, func);
    GELOGI("Register tiling function: op_type:%s, funcPointer:%p, registered count:%zu",
           op_type.c_str(), func.target<OpTilingFuncPtr>(), interf.size());
}

bool FeedTeOpTensorArg(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_desc, std::vector<TeOpTensorArg> &tensor_arg)
{
    for (auto &desc : tensor_desc) {
        TeOpTensorArg arg_tensor;
        TeOpTensor tensor;
        arg_tensor.arg_type = TA_SINGLE;
        tensor.shape = desc->GetShape().GetDims();
        tensor.ori_shape = desc->GetOriginShape().GetDims();

        tensor.format = ge::TypeUtils::FormatToSerialString(desc->GetFormat());

        tensor.ori_format = ge::TypeUtils::FormatToSerialString(desc->GetOriginFormat());

        ge::DataType dtype = desc->GetDataType();
        auto dataTypeIter = DATATYPE_STRING_MAP.find(dtype);
        if (dataTypeIter == DATATYPE_STRING_MAP.end()) {
            GE_LOGE("datatype error %d", static_cast<int>(dtype));
            return false;
        }
        tensor.dtype = dataTypeIter->second;
        if (LOG_ENABLED(DLOG_INFO)) {
            std::stringstream shapestr;
            shapestr << "shape:[";
            for (auto &i : tensor.shape) {
                shapestr << i << ",";
            }
            shapestr << "], ori_shape:[";
            for (auto &i : tensor.ori_shape) {
                shapestr << i << ",";
            }
            shapestr << "], format:" << tensor.format;
            shapestr << ", ori_format:" << tensor.ori_format;
            shapestr << ", dtype: " << tensor.dtype;
            GELOGI("calling optiling shape info: %s", shapestr.str().c_str());
        }

        arg_tensor.tensor.emplace_back(tensor);
        tensor_arg.emplace_back(arg_tensor);
    }
    return true;
}

void FeedTeOpConstTensor(const ge::Node &node, const ge::OpDescPtr &op_desc,
                         std::map<std::string, TeConstTensorData> &const_inputs)
{
    ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());
    std::vector<std::string> inferDepends = op_desc->GetOpInferDepends();

    for (auto &depend : inferDepends) {
        ge::Tensor data;
        ge::graphStatus rc = op.GetInputConstData(depend, data);
        GELOGI("GetInputConstData: %s, %d", depend.c_str(), rc);
        if (rc != ge::GRAPH_SUCCESS) {
            continue;
        }

        const uint8_t *pbuf = data.GetData();
        size_t buflen = data.GetSize();

        GELOGI("Const input tensor data: %s, %p %zu", depend.c_str(), pbuf, buflen);
        const_inputs.emplace(depend, TeConstTensorData{pbuf, buflen, data});
    }
}


bool GetCompileInfo(const ge::OpDescPtr &op_desc, nlohmann::json &dummy, nlohmann::json *&pJson,
                    const char *op_type, const char *op_name)
{
    int64_t compile_info_key = 0;
    std::string compile_info_uuid;
    std::string &uuid = OpTilingInterf::OpTilingUuid;

    bool bres = ge::AttrUtils::GetStr(op_desc, "compile_info_uuid", compile_info_uuid);
    if (!bres) {
        GELOGI("Can not found compile_info_uuid, op_type:%s, op_name:%s", op_type, op_name);
        // This op may has any compile_info
        pJson = &dummy;
        return true;
    }

    if (uuid == compile_info_uuid) {
        // Same process, use json pointer
        bres = ge::AttrUtils::GetInt(op_desc, "compile_info", compile_info_key);
        if (!bres) {
            GE_LOGE("Failed to get compile_info_uuid, op_type:%s, op_name:%s", op_type, op_name);
            return false;
        }
        GELOGI("Get op_type %d, %s, op_name:%s, compile_info_key:%ld, uuid:%s",
               static_cast<int>(bres), op_type, op_name, compile_info_key, uuid.c_str());
        pJson = reinterpret_cast<nlohmann::json*>(compile_info_key);
        if (pJson == nullptr) {
            pJson = &dummy;
        }
    } else {
        // Not in same process, need to parse compile_info json
        std::string compile_info_json;
        bres = ge::AttrUtils::GetStr(op_desc, "compile_info_json", compile_info_json);
        if (!bres) {
            GE_LOGE("Failed to get compile_info_json, op_type:%s, op_name:%s", op_type, op_name);
            return false;
        }
        GELOGI("Get op_type %d, %s, op_name:%s, compile_info_json:%s, uuid:%s/%s",
               static_cast<int>(bres), op_type, op_name, compile_info_json.c_str(),
               uuid.c_str(), compile_info_uuid.c_str());
        try {
            dummy = nlohmann::json::parse(compile_info_json);
            pJson = &dummy;
        } catch (...) {
            GE_LOGE("Failed to parse compile_info_json %s, op_type:%s, op_name:%s",
                    compile_info_json.c_str(), op_type, op_name);
            return false;
        }
    }
    return true;
}


extern "C" ge::graphStatus OpParaCalculate(const ge::Node &node, OpRunInfo &runInfo)
{
    TeOpParas op_param;
    ge::OpDescPtr op_desc = node.GetOpDesc();
    std::string op_type = op_desc->GetType();
    std::string op_name = op_desc->GetName();

    GELOGI("Do optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());

    auto inputs =  op_desc->GetAllInputsDescPtr();
    auto outputs = op_desc->GetAllOutputsDescPtr();

    bool bres = false;
    bres = FeedTeOpTensorArg(inputs, op_param.inputs);
    if (!bres) {
        GE_LOGE("Do optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }
    bres = FeedTeOpTensorArg(outputs, op_param.outputs);
    if (!bres) {
        return ge::GRAPH_FAILED;
    }

    FeedTeOpConstTensor(node, op_desc, op_param.const_inputs);

    auto &interf = OpTilingInterf::RegisteredOpInterf();
    nlohmann::json dummy;
    nlohmann::json *pJson = nullptr;
    bres = GetCompileInfo(op_desc, dummy, pJson, op_type.c_str(), op_name.c_str());
    if (!bres) {
        GE_LOGE("Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    auto iter = interf.find(op_type);
    if (iter == interf.end()) {
        iter = interf.find("AutoTiling");
    }

    if (iter != interf.end()) {
        TeOpParas opParas;
        GELOGI("Optiling func found, op_type:%s, op_name:%s, func:[%s:%p]",
               op_type.c_str(), op_name.c_str(), iter->first.c_str(), iter->second.target<OpTilingFuncPtr>());
        bool rc =  (iter->second)(op_type, op_param, *pJson, runInfo);
        if (rc) {
            GELOGI("Optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        } else {
            GE_LOGE("Optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        }
        return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }

    GE_LOGE("Optiling func not found, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
}

extern "C" ge::graphStatus OpAtomicCalculate(const ge::Node &node, OpRunInfo &runInfo)
{
    TeOpParas op_param;
    ge::OpDescPtr op_desc = node.GetOpDesc();
    std::string op_type = "DynamicAtomicAddrClean";
    std::string op_name = op_desc->GetName();
    std::string origin_op_type = "DynamicAtomicAddrClean";

    GELOGI("Do Atomic optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    std::vector<int64_t> atomic_output_indices;
    (void) ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
    if (atomic_output_indices.empty()) {
        GE_LOGE("No ATOMIC_ATTR_OUTPUT_INDEX found, op_type:%s, op_name:%s",
                origin_op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    auto tensor = op_desc->MutableOutputDesc(atomic_output_indices[0]);
    if (tensor == nullptr) {
        GE_LOGE("Get MutableOutputDesc failed. op_type:%s, op_name:%s",
                origin_op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t clean_size = 0;
    auto res = ge::TensorUtils::GetSize(*tensor, clean_size);
    if (res != ge::GRAPH_SUCCESS) {
        GE_LOGE("Get size of tensor desc failed. op_type:%s, op_name:%s",
                origin_op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    GELOGI("Atomic clean size: %ld, op_type:%s, op_name:%s",
           clean_size, origin_op_type.c_str(), op_name.c_str());
    op_param.const_inputs.emplace("workspace_size",
                                  TeConstTensorData(nullptr, static_cast<size_t>(clean_size), ge::Tensor()));

    auto &interf = OpTilingInterf::RegisteredOpInterf();
    auto iter = interf.find(op_type);
    if (iter == interf.end()) {
        GE_LOGE("Atomic op tiling func not found. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    ge::NodePtr atomic_clean_node = nullptr;
    atomic_clean_node = op_desc->TryGetExtAttr("atomic_clean_node_ptr", atomic_clean_node);
    if (atomic_clean_node == nullptr) {
        GE_LOGE("This node has no atomice node. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    ge::OpDescPtr atomic_op_desc = atomic_clean_node->GetOpDesc();
    if (atomic_op_desc == nullptr) {
        GE_LOGE("Failed to get op desc from node. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }

    nlohmann::json dummy;
    nlohmann::json *pJson = nullptr;
    bool bres = GetCompileInfo(atomic_op_desc, dummy, pJson, op_type.c_str(), op_name.c_str());
    if (!bres) {
        GE_LOGE("Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
        return ge::GRAPH_FAILED;
    }
    bool rc =  (iter->second)(op_type, op_param, *pJson, runInfo);
    if (rc) {
        GELOGI("Atomic optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    } else {
        GE_LOGE("Atomic optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    }

    return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}


