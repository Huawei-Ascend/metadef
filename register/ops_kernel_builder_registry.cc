/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "register/ops_kernel_builder_registry.h"

namespace ge {
void OpsKernelBuilderRegistry::Register(const string &lib_name, const OpsKernelBuilderPtr &instance) {
  kernel_builders_.emplace(lib_name, instance);
}

void OpsKernelBuilderRegistry::UnregisterAll() {
  kernel_builders_.clear();
}

const std::map<std::string, OpsKernelBuilderPtr> &OpsKernelBuilderRegistry::GetAll() const {
  return kernel_builders_;
}
OpsKernelBuilderRegistry &OpsKernelBuilderRegistry::GetInstance() {
  static OpsKernelBuilderRegistry instance;
  return instance;
}

OpsKernelBuilderRegistrar::OpsKernelBuilderRegistrar(const string &kernel_lib_name,
                                                     OpsKernelBuilderRegistrar::CreateFn fn) {
  auto instance = std::shared_ptr<OpsKernelBuilder>(fn());
  OpsKernelBuilderRegistry::GetInstance().Register(kernel_lib_name, instance);
}
}  // namespace ge