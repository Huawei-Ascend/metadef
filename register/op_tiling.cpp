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

#include "register/op_tiling.h"

#include <chrono>
#include <cstring>
#include "securec.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"


#define LOG_ENABLED(loglvl) CheckLogLevel(GE_MODULE_NAME, loglvl)

namespace optiling {
const char* COMPILE_INFO_JSON = "compile_info_json";
const char* COMPILE_INFO_KEY = "compile_info_key";

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

bool GetCompileInfo(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name,
    OpCompileInfo &op_compile_info) {
    std::string compile_info_key;
    bool bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
    if (!bres) {
        GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", COMPILE_INFO_KEY, op_type, op_name);
        return false;
    }

    std::string compile_info_str;
    bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, compile_info_str);
    if (!bres) {
        GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", COMPILE_INFO_JSON, op_type, op_name);
        return false;
    }
    op_compile_info.key = compile_info_key;
    op_compile_info.str = compile_info_str;
    return true;
}

bool RunCalcFunc(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name, const TeOpParas &op_param,
                 OpRunInfo & run_info) {
    auto &interf = OpTilingRegistryInterf::RegisteredOpInterfNew();
    auto iter = interf.find(op_type);
    if (iter == interf.end()) {
        iter = interf.find("AutoTiling");
    }

    if (iter == interf.end()) {
        GE_LOGE("Optiling func not found. op_type:%s", op_type);
        return false;
    }

    OpCompileInfo op_compile_info;
    bool res = GetCompileInfo(op_desc, op_type, op_name, op_compile_info);
    if (!res) {
        GE_LOGE("Failed to get compile info. op_type:%s, op_name:%s", op_type, op_name);
        return false;
    }

    GELOGI("Optiling func found. op_type:%s, op_name:%s, func:[%s:%p]",
           op_type, op_name, iter->first.c_str(), iter->second.target<OpTilingFuncPtrNew>());
    res = (iter->second)(op_param, op_compile_info, run_info);
    if (res) {
        GELOGI("Optiling func succeed. op_type:%s, op_name:%s", op_type, op_name);
    } else {
        GE_LOGE("Optiling func failed. op_type:%s, op_name:%s", op_type, op_name);
    }
    return res;
}

bool RunAtomicFunc(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name, const TeOpParas &op_param,
                 OpRunInfo & run_info) {
    auto &interf = OpTilingRegistryInterf::RegisteredOpInterfNew();
    auto iter = interf.find(op_type);
    if (iter == interf.end()) {
        GE_LOGE("Atomic optiling func not found. op_type:%s", op_type);
        return false;
    }

    ge::NodePtr atomic_clean_node = nullptr;
    atomic_clean_node = op_desc->TryGetExtAttr("atomic_clean_node_ptr", atomic_clean_node);
    if (atomic_clean_node == nullptr) {
        GE_LOGE("This node has no atomic node. op_type:%s, op_name:%s", op_type, op_name);
        return false;
    }

    ge::OpDescPtr atomic_op_desc = atomic_clean_node->GetOpDesc();
    if (atomic_op_desc == nullptr) {
        GE_LOGE("Failed to get op desc from node. op_type:%s, op_name:%s", op_type, op_name);
        return false;
    }

    OpCompileInfo op_compile_info;
    bool res = GetCompileInfo(atomic_op_desc, op_type, op_name, op_compile_info);
    if (!res) {
        GE_LOGE("Failed to get compile info. op_type:%s, op_name:%s", op_type, op_name);
        return false;
    }

    res = (iter->second)(op_param, op_compile_info, run_info);
    if (res) {
        GELOGI("Atomic optiling func succeed. op_type:%s, op_name:%s", op_type, op_name);
    } else {
        GE_LOGE("Atomic optiling func failed. op_type:%s, op_name:%s", op_type, op_name);
    }
    return res;
}

void ParseShapeDesc(const nlohmann::json &shape, std::vector<TeOpTensor> &tensors)
{
    TeOpTensor tensor;
    if (shape.contains("shape")) {
        tensor.shape = shape["shape"].get<vector<int64_t>>();
    }
    if (shape.contains("ori_shape")) {
        tensor.ori_shape = shape["ori_shape"].get<vector<int64_t>>();
    }
    if (shape.contains("format")) {
        tensor.format = shape["format"].get<std::string>();
    }
    if (shape.contains("ori_format")) {
        tensor.ori_format = shape["ori_format"].get<std::string>();
    }
    if (shape.contains("dtype")) {
        tensor.dtype = shape["dtype"].get<std::string>();
    }
    tensors.emplace_back(tensor);
}

void ParseShapeDescList(const nlohmann::json &shape_list, std::vector<TeOpTensorArg> &op_args)
{
    for (const auto &elem : shape_list) {
        TeOpTensorArg tensor_arg;
        tensor_arg.arg_type = TA_NONE;

        if (elem.is_array()) {
            tensor_arg.arg_type = TA_LIST;
            for (const auto &shape : elem) {
                ParseShapeDesc(shape, tensor_arg.tensor);
            }
        } else {
            tensor_arg.arg_type = TA_SINGLE;
            ParseShapeDesc(elem, tensor_arg.tensor);
        }
        op_args.emplace_back(tensor_arg);
    }
}

std::string DumpByteBuffer(const ByteBuffer &buf)
{
    static const char hex_digits[] = "0123456789ABCDEF";
    std::string str = buf.str();
    std::string output;
    output.reserve(str.size() * 2);
    for (unsigned char c : str) {
        output.push_back(hex_digits[c >> 4]);
        output.push_back(hex_digits[c & 15]);
    }
    return output;
}

bool DumpRunInfo(const OpRunInfo &run_info, char *run_info_json, size_t run_info_len)
{
    if (run_info_json == nullptr) {
        GE_LOGE("run_info buffer is null");
        return false;
    }

    nlohmann::json json_obj;
    json_obj["block_dim"] = run_info.block_dim;
    json_obj["workspaces"] = run_info.workspaces;
    json_obj["tiling_data"] = DumpByteBuffer(run_info.tiling_data);

    std::string str = json_obj.dump();
    if (str.size() >= run_info_len) {
        GE_LOGE("runinfo too large. %zu/%zu", str.size(), run_info_len);
        return false;
    }
    auto rc = memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1);
    if (rc != EOK) {
        return false;
    }
    return true;
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info,
                                  const char *inputs, const char *outputs,
                                  char *run_info_json, size_t run_info_len)
{
    if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
        GE_LOGE("optype/compile_info/inputs/outputs is null, %s, %s, %s, %s",
                optype, compile_info, inputs, outputs);
        return 0;
    }

    std::string compile_info_str = compile_info;
    TeOpParas op_params;
    nlohmann::json compile_info_json;
    try {
        compile_info_json = nlohmann::json::parse(compile_info);
        nlohmann::json inputs_json = nlohmann::json::parse(inputs);
        nlohmann::json outputs_json = nlohmann::json::parse(outputs);
        ParseShapeDescList(inputs_json, op_params.inputs);
        ParseShapeDescList(outputs_json, op_params.outputs);
    } catch (...) {
        GE_LOGE("Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
        return 0;
    }

    auto &interf = OpTilingInterf::RegisteredOpInterf();
    auto iter = interf.find(optype);
    if (iter == interf.end()) {
        iter = interf.find("AutoTiling");
    }
    
    if (iter == interf.end()) {
        GE_LOGE("Optiling func not found, op_type:%s", optype);
        return 0;
    }

    OpRunInfo run_info;
    GELOGI("Optiling func found, op_type:%s, func:[%s:%p]",
           optype, iter->first.c_str(), iter->second.target<OpTilingFuncPtr>());

    bool rc = (iter->second)(optype, op_params, compile_info_json, run_info);
    if (!rc) {
        GE_LOGE("Optiling failed. op_type:%s", optype);
        return 0;
    }
    GELOGI("Optiling succeed. op_type:%s", optype);
    DumpRunInfo(run_info, run_info_json, run_info_len);
    return 1;
}

extern "C" ge::graphStatus OpParaCalculate(const ge::Node &node, OpRunInfo &runInfo)
{
    ge::OpDescPtr op_desc = node.GetOpDesc();
    std::string op_type = op_desc->GetType();
    std::string op_name = op_desc->GetName();
    TeOpParas op_param;
    op_param.op_type = op_type;

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

    if (!RunCalcFunc(op_desc, op_type.c_str(), op_name.c_str(), op_param, runInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus OpAtomicCalculate(const ge::Node &node, OpRunInfo &runInfo)
{
    ge::OpDescPtr op_desc = node.GetOpDesc();
    std::string op_type = "DynamicAtomicAddrClean";
    std::string op_name = op_desc->GetName();
    std::string origin_op_type = "DynamicAtomicAddrClean";
    TeOpParas op_param;
    op_param.op_type = op_type;

    GELOGI("Do Atomic optiling. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
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
        if (!RunAtomicFunc(op_desc, op_type.c_str(), op_name.c_str(), op_param, runInfo)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
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


