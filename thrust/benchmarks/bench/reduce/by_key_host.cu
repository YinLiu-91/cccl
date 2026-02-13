// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
//
// CPU 版本：使用 thrust::host_vector，在主机上执行 reduce_by_key，与 by_key.cu (GPU) 对照。
//
// 执行策略：thrust::host 由 THRUST_HOST_SYSTEM 决定；默认 CPP = 单线程。
// 若要多线程，请用 OpenMP 构建并选用 OMP 作为 host：
//   cmake -DTHRUST_HOST_SYSTEM=OMP ...   （需 FindOpenMP，并链接 OpenMP）
// 或启用 multiconfig 并打开 OMP：THRUST_ENABLE_MULTICONFIG=ON、THRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON，
// 会生成 thrust.omp.cuda 等目标，对应可执行文件即为多线程 CPU 版本。

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>

#include <cuda/std/iterator>

#include <nvbench/nvbench.cuh>

#include <random>

namespace detail
{
// 在主机上生成均匀段长的 key 序列，用于 reduce_by_key 测试
template <typename KeyT>
void gen_uniform_key_segments_host_simple(thrust::host_vector<KeyT>& keys,
                                          std::size_t total_elements,
                                          std::size_t min_segment_size,
                                          std::size_t max_segment_size,
                                          unsigned seed = 42)
{
  keys.resize(total_elements);
  std::default_random_engine rng(seed);
  std::uniform_int_distribution<std::size_t> seg_dist(min_segment_size, max_segment_size);

  KeyT key_val = 0;
  std::size_t pos = 0;
  while (pos < total_elements)
  {
    std::size_t seg_len = seg_dist(rng);
    if (pos + seg_len > total_elements)
      seg_len = total_elements - pos;
    for (std::size_t i = 0; i < seg_len; ++i)
      keys[pos + i] = key_val;
    pos += seg_len;
    ++key_val;
  }
}
} // namespace detail

template <class KeyT, class ValueT>
static void basic_host(nvbench::state& state, nvbench::type_list<KeyT, ValueT>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::host_vector<KeyT> in_keys;
  detail::gen_uniform_key_segments_host_simple(in_keys, elements, min_segment_size, max_segment_size);

  thrust::host_vector<KeyT> out_keys = in_keys;
  thrust::host_vector<ValueT> in_vals(elements);
  thrust::fill(in_vals.begin(), in_vals.end(), static_cast<ValueT>(1));

  const std::size_t unique_keys = static_cast<std::size_t>(
    ::cuda::std::distance(out_keys.begin(), thrust::unique(out_keys.begin(), out_keys.end())));

  thrust::host_vector<ValueT> out_vals(unique_keys);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(unique_keys);
  state.add_global_memory_writes<ValueT>(unique_keys);

  state.exec(
    nvbench::exec_tag::no_gpu, [&](nvbench::launch&) {
      thrust::reduce_by_key(thrust::host,
                            in_keys.begin(),
                            in_keys.end(),
                            in_vals.begin(),
                            out_keys.begin(),
                            out_vals.begin());
    });
}

using key_types =
  nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

using value_types_host = nvbench::type_list<int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(basic_host, NVBENCH_TYPE_AXES(key_types, value_types_host))
  .set_name("base_host")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .set_is_cpu_only(true)
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
