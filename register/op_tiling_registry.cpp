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
#include "register/op_tiling_registry.h"

#include <random>
#include "framework/common/debug/ge_log.h"

namespace optiling {

constexpr int UUID_LENGTH = 32;

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

std::map<std::string, OpTilingFuncNew> & OpTilingRegistryInterf::RegisteredOpInterfNew() {
    static std::map<std::string, OpTilingFuncNew> interf_new;
    return interf_new;
}

OpTilingRegistryInterf::OpTilingRegistryInterf(std::string op_type, OpTilingFuncNew func) {
    auto &interf_new = RegisteredOpInterfNew();
    interf_new.emplace(op_type, func);
    GELOGI("Register tiling function: op_type:%s, funcPointer:%p, registered count:%zu",
           op_type.c_str(), func.target<OpTilingFuncPtrNew>(), interf_new.size());
}
}
