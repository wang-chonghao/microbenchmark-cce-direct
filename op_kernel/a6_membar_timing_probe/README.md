# A6 Membar 一键测试

本目录原样导入 `VfSimulator/cce_code/a6_membar_timing_probe` 的四个 CCE、公共头和提取器，原说明保留为 `UPSTREAM_README.md`，导入哈希见 `import_provenance.json`。远程无需原仓库。

## 一键运行

在 microbenchmark-cce-direct 根目录使用现有双 CANN 环境：

```bash
export CANN_HOME=/your/compiler/cann
export FULL_SIMULATOR_HOME=/your/debug/cann/tools/simulator
export SOC_VERSION=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
source set_env.sh
python run_a6_membar.py --output results/a6_membar_run1
```

这一个命令会依次编译、直接运行 CAmodel、做精确字节校验、归档日志并提取参数。默认 device=0、block_dim=1、每个 kernel 执行一次、无超时。输出目录必须不存在。单例失败后继续其余用例，最终退出码非零；不会把失败运行的日志当成有效测量。

```bash
# 列出四例
python run_a6_membar.py --list

# 只跑一例
python run_a6_membar.py --cases 03_interleave_store --output results/a6_membar_store

# 只编译设备 kernel，不依赖 A6 模型或 host 链接库
python run_a6_membar.py --compile-only --output results/a6_membar_compile
```

`--timeout 3600` 可限制每例编译+模型运行的总秒数；默认 `0` 无限等待。`--cann-home`、`--full-simulator-home` 可替代环境变量。

## 编译与正确性

| 用例 | 目标屏障 | 用途 |
|---|---|---|
| 01_cluster_vld_vst | 连续 8 条 VLD_VST | 发射→放行、发射→退休、退休→下一屏障发射 |
| 02_cluster_vst_vld | 连续 8 条 VST_VLD | 同上，另一方向 |
| 03_interleave_store | 8 组 VLD_VST→STORE | 屏障放行→后续 STORE 开始 |
| 04_interleave_load | 8 组 VST_VLD→LOAD | 屏障放行→后续 LOAD 开始 |

02/04 还含两条辅助 VLD_VST，提取器不会将其混入八条目标屏障。默认去掉目标屏障首尾各一条，用中间六条统计。

固定 A6 `dav-920r1-vec`，VF fusion 关闭、`vec-misched=0`、VF stack=0。**本套受控实验替换而不是继承 `CCEC_EXTRA_FLAGS`**，并打印提示，防止继承其他实验的架构或 spill 参数。不会因编译失败自动放大 VF 栈。run_test.py 的基础编译参数仍保留。

CCE 和公共头不改计算/同步，只在运行快照前加 MB_KERNEL/MB_TENSOR 注释，使已有通用 runner 能识别两个参数。完整 NDDMA/VF/scatter 均保留。

输入 64 个 FP32，输出 512 个 FP32。实际输出必须逐字节等于输入重复八份。使用现有 generic runner 的前后各 256B GM 保护区和输入不变性检查，因此不另在输出张量内增加原 host 的尾部 guard。保护区只能检测对应边界的写越界，不能证明 UB 内部完全无越界。

## 日志与结果

```text
results/a6_membar_run1/
  _work/<case>/kernel.cce
  _work/<case>/probe_common.h
  _work/<case>/build/
  _work/<case>/debug.log
  _work/<case>/model.log
  _work/<case>/console.log
  _work/<case>/receipt.json
  _work/<case>/golden.json
  logs/<case>/core0.veccore0.*.dump
  logs/<case>/raw/
  logs/<case>/archive.json
  measurement/overview.json
  measurement/<case>/barriers.csv
  measurement/<case>/summary.json
  measurement/<case>/README.md
```

**本测试不能仅保留原先六类日志。** 默认保留全部 core0.veccore0 的 dump，包括 SU/ISU、depmgr、LSU/GSU；其他 core 的 dump 在当前用例结束后删除。`--keep-all-dumps` 可保留其他 core 的原始 dump。不会删除历史结果、脚本或环境文件。

轮转文件及 `.dump.gz` 保存在 `raw/`，另拼接为提取器支持的标准文件名；`archive.json` 保存原文件 SHA256、原路径和归一化行号映射，便于回溯证据。同名日志来自多个目录时拒绝合并，避免混入不同执行。

提取器检查 VF 完整性、目标屏障数量、LOAD/STORE 数量、是否出现循环/额外计算/spill；编译 receipt 同时核对本次源码、公共头及二进制哈希。确认 golden 和这些条件后才输出观测候选。

`overview.json/directions` 汇总两方向的 `release_latency`、`retire_latency`、`next_issue_delay` 和 `consumer_delay`。详细 I/G/R/C、反馈间隔上界和控制阶段证据见各 case 的 CSV/JSON。重复相同差值至少三次才给稳定候选，否则为 null；**所有结果仍是 calibrated=false**，不自动写入仿真器架构配置。`admission_delay` 无法独立辨识，保持 null。

没有详细事件时标为 `timing_incomplete`，保留已有证据并返回非零状态。解析器没有识别到事件，不等于该延迟为零；popped 时间也不能替代 SU issue。A6 的事件格式需以你服务器实际日志为准。

## 不重跑模型，重新解析

如果 debug 包的 SU 字段不同，可以按 `event_patterns.example.json` 编写匹配规则，然后：

```bash
python run_a6_membar.py --analyze-only --output results/a6_membar_run1 \
  --patterns /path/to/a6_event_patterns.json --control-unit SU
```

默认写入新的 `measurement_<时间戳>/`，不覆盖旧提取结果。也可用 `--analysis-output` 指定尚不存在的目录。`--cases` 可只分析某例；`--vf-index` 用于显式选择 VF。重新解析会再次检查源码/二进制 receipt、输入输出及 guard，不接受未通过正确性验证的历史运行。

本地仅验证编译链接与解析测试；没有 A6 CAModel 库，未声称已测得 A6 参数。原说明中的 A5 历史数值不属于本次 A6 结果。
