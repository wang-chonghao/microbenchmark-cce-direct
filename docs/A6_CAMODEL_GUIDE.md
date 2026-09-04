# A6 CCE CAmodel 运行与数据搬运指南

本文记录 `microbenchmark-cce-direct` 在 A6 `Ascend910_9691` SoC 上直接运行
CCE kernel 的配置，以及已经通过 `A = B + C, fp32[256]` 精度验证的数据搬运写法。

这里不使用 `msprof op simulator` 或 `cannsim`。`run_test.py` 编译 CCE kernel 和
native runner，然后由 runner 通过 CANN runtime 启动 CAmodel。

## 1. 已验证范围

以下组合已经通过 A6 CAmodel 精度验证：

```text
SoC                 Ascend910_9691
CCE core arch       dav-920r1-vec
core simulator dir  dav_9201
kernel              A = B + C
dtype/shape         fp32[256]
GM -> UB             nddma_out_to_ub
UB -> GM             scatter_ubuf_to_gm
scatter mode         A6_SCATTER_MODE=0
offset type          A6_OFFSET_TYPE=0
```

对应代码是：

```text
op_kernel/a6_scatter_probe.cce
```

注意：`op_kernel/kernel.cce` 当前使用四参数 V920
`copy_ubuf_to_gm_align_v2` 作为 A6 CopyOut。该形式已经通过 A6 编译检查，但尚未记录
A6 CAmodel 精度验证结果。需要稳定复现时，优先使用已经验证的 scatter probe。

## 2. 环境配置

进入仓库后，将路径替换为当前机器的实际安装路径：

```bash
cd /path/to/microbenchmark-cce-direct

export CANN_HOME=/absolute/path/to/cann
export ARCH=x86_64-linux
export SOC_VERSION=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201

source ./set_env.sh
```

如果编译/runtime CANN 包和全量日志 simulator 包分开安装，再设置：

```bash
export FULL_SIMULATOR_HOME=/absolute/path/to/full-dump/tools/simulator
source ./set_env.sh
```

`FULL_SIMULATOR_HOME` 应当直接指向包含 SoC/core simulator 子目录的
`tools/simulator` 目录。例如：

```text
$FULL_SIMULATOR_HOME/Ascend910_9691/
$FULL_SIMULATOR_HOME/dav_9201/
```

检查最终环境：

```bash
printf 'CANN_HOME=%s\nSOC_VERSION=%s\nCORE_ARCH=%s\nCORE_SIM_DIR=%s\nFULL_SIMULATOR_HOME=%s\n' \
  "$CANN_HOME" "$SOC_VERSION" "$CORE_ARCH" "$CORE_SIM_DIR" "$FULL_SIMULATOR_HOME"
```

关键值必须是：

```text
SOC_VERSION=Ascend910_9691
CORE_ARCH=dav-920r1-vec
CORE_SIM_DIR=dav_9201
```

## 3. 运行已验证的 A6 用例

执行：

```bash
CCEC_EXTRA_FLAGS="-DA6_SCATTER_MODE=0 -DA6_OFFSET_TYPE=0" \
python3 run_test.py \
  --kernel op_kernel/a6_scatter_probe.cce \
  --output result/a6_scatter_m0_t0
```

成功时应当看到：

```text
[CHECK] PASS A: B + C
```

结果目录结构：

```text
result/a6_scatter_m0_t0/
  build/add_simd_a6_scatter_probe_mix_aiv.o
  build/add_simd_a6_scatter_probe_mix.o
  debug.log
  model.log
  run/input_B.bin
  run/input_C.bin
  run/output_A.bin
  run/golden_check.log
  run/log_ca/*.dump
```

确认实际加载的是目标 simulator：

```bash
grep -A80 "native runner linked simulator libs" \
  result/a6_scatter_m0_t0/debug.log
```

其中 `libruntime_camodel.so`、`libnpu_drv_camodel.so`、`libpem_davinci.so` 等库
应来自同一套 A6 simulator，不能混用 A5 `dav_3510` 库。

## 4. A6 GM 到 UB：NDDMA

fp32 连续输入的已验证写法如下：

```cpp
constexpr uint32_t elementCount = 256;

__ubuf__ float* ubB = (__ubuf__ float*)get_imm(0x0000);
__gm__ float* B = /* kernel argument */;

nddma_desc copyInDesc(nddma_desc::loop_desc(elementCount, 1, 1));
nddma_out_to_ub(
    ubB,                // UB destination
    B,                  // GM source
    0,                  // sid
    copyInDesc,         // NDDMA loop descriptor
    0,                  // padding value
    NEAREST_PADDING,    // padding mode
    0);                 // L2 cache control
```

本例中的 descriptor：

```cpp
nddma_desc::loop_desc(elementCount, 1, 1)
```

三个参数依次对应当前 loop 的：

```text
size        = 256
dst_stride  = 1
src_stride  = 1
```

对于 typed fp32 NDDMA，这组参数表示连续搬运 256 个 fp32 元素。两个输入可以共用
同一个只读 descriptor：

```cpp
nddma_out_to_ub(ubB, B, 0, copyInDesc, 0, NEAREST_PADDING, 0);
nddma_out_to_ub(ubC, C, 0, copyInDesc, 0, NEAREST_PADDING, 0);
```

NDDMA 完成后，Vector pipeline 读取 UB 前必须同步：

```cpp
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

## 5. A6 UB 到 GM：Scatter

### 5.1 已验证参数

对于 `ubA` 中连续的 256 个 fp32，即 1024 bytes，使用一个 offset 描述一个连续
burst：

```cpp
constexpr uint32_t dataBytes = 256 * sizeof(float);

__ubuf__ float* ubA = (__ubuf__ float*)get_imm(0x2000);
__ubuf__ uint32_t* offsets = (__ubuf__ uint32_t*)get_imm(0x3000);

offsets[0] = 0;

constexpr uint32_t burstLen = dataBytes; // 1024 bytes
constexpr uint32_t offsetNum = 1;
```

Scatter 调用：

```cpp
scatter_ubuf_to_gm(
    reinterpret_cast<__gm__ uint32_t*>(A),
    reinterpret_cast<__ubuf__ uint32_t*>(ubA),
    0,                  // sid
    0,                  // offset_type: verified value for this uint32_t offset table
    offsets,            // offsets[0] = 0
    0,                  // L2 cache control
    burstLen,           // 1024 bytes
    0,                  // source gap/stride for contiguous source data
    offsetNum);         // one offset entry
```

关键点：

- `src_stride` 必须为 `0`，表示相邻 burst 之间没有额外源间隔。
- 不要把 `dataBytes` 填入 `src_stride`。
- `offsets[0]=0` 表示相对于 GM 基地址 `A` 的起始 offset 为 0。
- `offset_num=1` 表示 offset 表只有一个有效项。
- fp32 数据按位转换为 `uint32_t*`，不会进行数值类型转换。

### 5.2 必须保留的同步

`ubA` 由 Vector pipeline 写入，offset 表由 Scalar pipeline 写入，而 scatter 由
MTE3 消费。因此已验证用例保留以下同步：

```cpp
pipe_barrier(PIPE_ALL);

set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

scatter_ubuf_to_gm(
    reinterpret_cast<__gm__ uint32_t*>(A),
    reinterpret_cast<__ubuf__ uint32_t*>(ubA),
    0, 0, offsets, 0, dataBytes, 0, 1);

set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
```

最后一组同步保证 CopyOut 完成后 kernel 才结束。在重新做 CAmodel 精度验证之前，
不要删除或合并这些同步。

## 6. 完整数据流

```text
GM(B/C)
  -> nddma_out_to_ub
  -> PIPE_MTE2 -> PIPE_V synchronization
  -> UB(ubB/ubC)
  -> vlds / vadd / vsts
  -> UB(ubA)
  -> PIPE_V and PIPE_S -> PIPE_MTE3 synchronization
  -> scatter_ubuf_to_gm
  -> PIPE_MTE3 -> PIPE_S synchronization
  -> GM(A)
```

## 7. 精度失败排查

Runner 在运行 kernel 前使用 `0xCD` 初始化输出 buffer。如果出现：

```text
[CHECK] FAIL A[0] actual=-431602080 expected=2.380000055
```

`-431602080` 正好是 bit pattern `0xCDCDCDCD` 按 fp32 解释后的值。这表示输出 GM
仍保持 runner 的初始化值，即 CopyOut 没有写入 `A`，不是普通浮点精度误差。

可以确认输出内容：

```bash
xxd -g 4 -l 64 result/<run>/run/output_A.bin
```

如果持续看到 `cdcdcdcd`，依次检查：

1. 是否运行 `op_kernel/a6_scatter_probe.cce`。
2. 是否设置 `A6_SCATTER_MODE=0` 和 `A6_OFFSET_TYPE=0`。
3. Scatter 的 `src_stride` 是否为 `0`。
4. `offsets[0]` 是否在 scatter 前写为 `0`。
5. V/S 到 MTE3、MTE3 到 S 的同步是否完整。
6. 编译器、runtime 和 simulator 是否属于兼容的 A6 版本。

## 8. 修改 shape 时

修改 fp32 元素数量时，至少同步调整：

```cpp
elementCount
dataBytes = elementCount * sizeof(float)
nddma_desc::loop_desc(elementCount, 1, 1)
MB_TENSOR 中的元素数量
```

单 offset scatter 形式还要求目标数据在 UB 和 GM 中均为连续布局：

```cpp
offsets[0] = 0;
burstLen = dataBytes;
src_stride = 0;
offset_num = 1;
```

改变 dtype、非连续布局、多个 burst 或多维 descriptor 后，应重新执行 golden 精度
验证，不能直接沿用本例参数。
