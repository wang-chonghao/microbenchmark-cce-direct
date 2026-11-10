# A6 Softmax 多 Shape 批量实验

来源：`a6_softmax_fusion_results_summary.html` 中的 `runnable_a6`，八份完整 CCE 按 SHA256 原样导入。源 VF 的内部同步没有增加或删除。

| Shape | no_membar | three_loop |
|---|---|---|
| 8 x 1024 | 两行错位融合 | 三循环基线 |
| 12 x 1024 | 两行错位融合 | 三循环基线 |
| 16 x 1024 | 两行错位融合 | 三循环基线 |
| 24 x 1024 | 两行错位融合 | 三循环基线 |

输入 FP32，输出 FP16，不做 attention scale。`no_membar` 没有 VF 内部 barrier；`three_loop` 保留两条 `mem_bar(VST_VLD)`。外层均使用 NDDMA GM→UB、VF 运算、scatter UB→GM，包含外层流水同步。外层同步不能保证无 barrier VF 的内部 UB 依赖正确，因此两个版本都必须实际精度检查。

## 环境与执行

在仓库根目录执行，路径换成远程机器的真实安装位置：

```bash
export CANN_HOME=/your/compiler/cann
export FULL_SIMULATOR_HOME=/your/debug/cann/tools/simulator
export SOC_VERSION=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
source set_env.sh
python run_a6_operators.py --list
python run_a6_operators.py --output results/a6_softmax_shapes_run1
```

编译来自 `CANN_HOME`；模拟器由已有 runner 按 `FULL_SIMULATOR_HOME` 选择。直接执行 native runner，不使用 msprof。请确认 `debug.log` 中实际动态库来源。

先测试小 shape：

```bash
python run_a6_operators.py --rows 8 --output results/a6_softmax_smoke
```

只编译设备 kernel（不要求本机安装 A6 模型）：

```bash
python run_a6_operators.py --compile-only --output results/a6_softmax_compile
```

默认按 shape 从小到大串行执行，单例失败继续后续用例，最终退出码非零。`--timeout 0` 默认无限等待；可用 `--timeout 3600` 设置单例编译和运行合计时限。输出目录必须不存在，禁止复用旧结果。用 `--rows` 和 `--variants` 选择重跑范围；目前没有自动断点续跑。

## 输出与校验

- `summary.json`：各例运行状态、数值校验和日志完整性；不会把预测周期标作实测。
- `_work/<case>/`：源代码快照、输入、二进制、console/debug/model 日志。
- `_work/<case>/run/`：实际输入输出、GM 保护区检查。
- `dumps/softmax/<case>/`：core0.veccore0 的 instr_popped_log、instr_log、EXU、IDU、ISU、OOO；文件名带 shape 和版本。

默认运行结束后删除该新运行中其他 dump，沿用单指令实验归档规则。`--keep-lsu-dumps` 增加 LSU；`--keep-all-dumps` 保留其余原始 dump。缺失日志单独列在 `missing_or_empty`，精度通过不等于六类日志齐全。

输入使用固定种子的有限随机 FP32 数；相同 shape 的两个版本使用同一输入。独立 golden 按实际输入计算 FP64 稳定 softmax，检查全部 FP16 输出，容差 `abs <= 1e-6 OR rel <= 1e-3`。runner 同时检查 GM 前后保护区及只读输入未被改写；这不覆盖 UB 内部越界。`--seed` 可改变输入种子。

## 验证边界

本地只做 A6 编译/设备链接验证，没有 A6 CAmodel 数值通过结论。无内部 membar 的融合实现尤其需要目标模拟器验证。

32 x 1024 的完整 buffer 超过报告的 248 KiB UB 预算，未纳入可运行列表。切块会改变实验指令流，不以切块结果替代原报告的 32 行实验。24 行显式布局约 240 KiB，还需核对实际模型的 UB 容量及编译器额外栈空间。

此目录仅覆盖 softmax 专题八例，不代表 `spill_candidate_alignment_standalone.html` 的全部算子已经迁移。整算子批跑独立于 `run_a6_bench.py` 单指令 II/forwarding 测试。
