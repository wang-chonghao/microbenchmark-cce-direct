# A6 Spill 报告算子批跑

当前纳入 **100 例、24 类算子：68 个 original、32 个 software**。不是报告的全部版本：92 个报告行的原始版和 41 个 A6 优化版共 133 项，另有 33 项在 `manifest.json` 的 `excluded` 中列出路径及原因。

来源是 `spill_candidate_alignment_standalone.html` 的原始源码链接和 `software_cce_links.json` 的 **A6** 链接，不使用 A5 优化版，也不把反汇编 replay 当原始算子。导入时核对 standalone 内嵌索引的 metadata/source SHA256。远程运行只需要此仓库，不依赖原报告或 E 盘。

## 开始测试

在仓库根目录，使用与 softmax 测试相同的双 CANN 环境：

```bash
export CANN_HOME=/your/compiler/cann
export FULL_SIMULATOR_HOME=/your/full-debug/cann/tools/simulator
export SOC_VERSION=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
source set_env.sh
python run_a6_operators.py --suite spill --list
```

先跑一组 SwiGLU，包括 u1 原始版、u4 原始版、u4 优化版：

```bash
python run_a6_operators.py --suite spill --families swiglu --output results/a6_spill_smoke
```

跑全部 100 例：

```bash
python run_a6_operators.py --suite spill --output results/a6_spill_run1
```

按算子、版本或具体名称过滤：

```bash
python run_a6_operators.py --suite spill --families focal groupnorm logit --output results/a6_selected
python run_a6_operators.py --suite spill --roles software --output results/a6_software_only
python run_a6_operators.py --suite spill --names lstm-u7_original --output results/a6_lstm_retry
```

串行执行，避免多个全芯片模型争抢资源；默认 device 0、block dim 1、无限超时。`--timeout 3600` 设置每例编译与仿真总限时。单例失败后继续，最后返回非零状态。结果目录必须不存在；暂不支持自动 resume，可用 `--names` 指定失败用例重跑。

只编译/设备链接（无需 A6 模型库）：

```bash
python run_a6_operators.py --suite spill --compile-only --output results/a6_spill_compile
```

`--suite` 默认仍是 `softmax`，不会改变之前的 softmax 批跑命令。此脚本不是单指令 II/forwarding 参数提取器。

## 算子与文件

覆盖 swiglu、focal、swiglu_quant、bce、adam_decay、groupnorm、logit、lstm、syncbn、deepnorm、mish、logsigmoid_grad、binary_cross_entropy、groupnorm_swish_dx、logsigmoid、layernorm_dx、softplus、moe_softplus、causal_conv、rope_grad、rmsnorm_dx、softmax_ce、mhc_partial、compressor_softmax。

每个 `<报告行ID>_<original|software>/` 目录包含：

- `kernel.cce`：可供本仓库 runner 编译运行的 A6 版本。
- `original.cce`：报告指向的未经修改的源码快照，保留原版权声明。
- `input_input.bin`：归档输入原始字节，原始版和优化版使用相同数据。

`manifest.json` 记录源路径、源码/输入哈希、搬运大小、输出 dtype/区域、offset 表地址和单例编译参数。原始输入中可能有混合 FP16/FP32/整数布局，所以用 uint8 张量传输原始字节，不再用 host 的通用整数序列填充。

## A6 适配内容

仅迁移完整 byte-buffer wrapper：GM→UB 改为 NDDMA，UB→GM 改为 scatter（mode=0、offset type=0、单段输出）。保留 VF 内部的运算、循环和 membar。

VF 完成后，通过外层 V→S、S→MTE3 同步，使用不与任何输出区域重叠的 32B UB 空间存放 scatter offsets。输入或 VF 临时值此时已不再使用；不会在 VF 尚未完成时覆盖它们。最后等待 MTE3 搬出结束。外层同步不能代替 VF 内部依赖同步。

去掉未实际使用的 `kernel_operator.h` 引入，改用 stdint/aicore 宏；对依赖 `__local_mem__` 的源码，按 CANN 定义补 `__local_mem__ = __ubuf__`。原文件仍完整保留，可逐项比较。

沿用归档编译命令的 `-mllvm -cce-aicore-vec-misched=0`，避免让默认机器调度器重新安排这些源码调度实验。

`lstm-u7_original` 在初始编译检查中超过默认 6 KiB VF 栈（需要 7712B），仅此例附加 `-mllvm -cce-vf-stack-size=8192`，记录在 manifest 和结果中。这个差异需要在分析性能时考虑；没有修改所有用例的栈配置。其他额外参数仍来自 `CCEC_EXTRA_FLAGS`，不要遗留 A5 或错误 SoC 参数。

## 输出与正确性边界

```text
results/a6_spill_run1/
  summary.json
  comparisons.json
  manifest.json
  _work/<case>/kernel.cce
  _work/<case>/console.log
  _work/<case>/debug.log
  _work/<case>/model.log
  _work/<case>/run/input_input.bin
  _work/<case>/run/output_output.bin
  _work/<case>/run/guard_check.txt
  dumps/<family>/<case>/<case>/*dump
```

默认仅归档 core0.veccore0 的 instr_popped_log、instr_log、EXU、IDU、ISU、OOO，其他 dump 在本次运行结束后清理。`--keep-lsu-dumps` 增加 LSU，`--keep-all-dumps` 保留其他原始 dump。缺失或空日志记录在 summary 的 `missing_or_empty` 中。

`executed_unvalidated` 仅表示 runner 正常退出、GM 保护区/输入不变性检查通过、输出文件大小正确，**不表示算子数值正确**。100 例尚未实现独立算子 golden，也不拿 A5 运行时间作为 A6 实测。

同一批中成功执行的 original/software 会在 `comparisons.json` 做差分检查：确认输入哈希、输出布局一致；浮点使用 `abs <= 1e-5 + 1e-3 * abs(reference)`，int32 精确比较，非有限数判为不匹配。这只是两个实现间的有限输入对照，不证明独立正确性。只跑 software 时不会伪造对照结果。差分不匹配会使批跑最终退出码非零，但仍保留其他用例结果。

本机的 `compile_validation.json` 记录 100/100 通过 `dav-920r1-vec` 编译及设备链接；编译器为 CANN 9.0.0 的 clang 15.0.5（2026-04-25）。本机没有 A6 CAmodel，运行、实际 spill、UB 占用、内部同步及数值需要远程验证。重新编译可能改变报告中的寄存器分配和调度，因此预测周期不能直接作为验收值。

## 尚未纳入

33 个历史项包括 softmax_three_loop、gelu_poly、swiglu_grad、adam_apply_one、gelu_grad、gelu_grad_three_stage、flash_attention_grad_front、mhc_sinkhorn 的部分版本。它们使用不同 kernel 接口、直接 get_imm 输出或没有匹配的归档输入，需要分别恢复 host 数据布局。不是断言这些算子不支持 A6。

为避免制造错误用例，未用随机字节填补历史输入，也未把 VF-only 源码伪装成完整可运行算子。准确列表见 `manifest.json/excluded`。之前 softmax 多 shape 八例在独立目录 `a6_softmax_shapes`，不受这里的历史 softmax 缺项影响。

重建导入（仅维护者需要原报告目录；目标目录必须不存在）：

```bash
python -m benchmarks.a6.import_spill_report --report-root /path/to/membar_results --output /path/to/new_suite
```
