/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

std::map<std::string, OpTilingFuncNew> & OpTilingRegistryInterf::RegisteredOpInterf() {
    static std::map<std::string, OpTilingFuncNew> interf_new;
    return interf_new;
}

OpTilingRegistryInterf::OpTilingRegistryInterf(std::string op_type, OpTilingFuncNew func) {
    auto &interf_new = RegisteredOpInterf();
    interf_new.emplace(op_type, func);
    GELOGI("Register tiling function: op_type:%s, funcPointer:%p, registered count:%zu",
           op_type.c_str(), func.target<OpTilingFuncPtrNew>(), interf_new.size());
}
}
