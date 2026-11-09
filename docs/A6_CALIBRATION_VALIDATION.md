# A6 输入校验与校准实验

## 当前保证与限制

`ii_min`、`forwarding_min`、`latency_min` 均为日志中的观察间隔，**不是自动校准完成的硬件常数**。
每份 measurement 和 summary 现在都包含 `timing_quality`、`hardware_calibrated`，
目前后者始终为 false。缺少真实源就绪、前端供给、资源阻塞、饱和性证据时不会提升状态。
已有 `observed` 状态保持兼容，它只表示观察到了样本。

| 内容 | 已实现 | 尚未实现/证明 |
| --- | --- | --- |
| raw 输入 | 按语义 dtype 编码，再通过 uint32 传输；核对 runner 实际输入哈希 | FP4 编码没有经过审定，不用小整数位模式冒充合法输入 |
| 数值参考 | 常见逐元素运算、VAXPY、普通连续 load/store；校验每个 tile 全部有效 lane | 复杂转换、谓词、进位、特殊访存等仍为 not_implemented |
| 保护检查 | 所有 GM tensor 前后各 256 字节 canary；只读输入不变性 | UB 内部越界、未定义 lane、填充区域不作 canary 保证 |
| II | 同 EXU 独立指令观察间隔；可选 8/16 独立流对照 | ISU source-ready/stall 格式尚未解析，不能证明流水线已饱和 |
| forwarding | 按实际 operand role 分类；可选 Vn、Vm、旧 Vd 链 | popped 间隔不等于旁路可用时刻；非目标源就绪仍未证明 |
| 跨单元 | VLDS→VADD、VADD→VSTS 两个 FP32 RAW 探针 | 不是全指令配对矩阵，也没有实现跨指令 II 校准 |
| LSU | 可只额外归档 LSU，排除跨端口/重复请求指令 | 剩余值是指定 LSU 请求资源间隔，不是完整访存指令 II |

基础 787 个源码保持不变。按当前参考实现统计，759 个用例有已支持的语义输入编码，
其中 245 个有数值参考；其余 514 个继续明确报告 `golden=not_implemented`。
28 个 FP4 用例因尚无审定的输入编码，会在模型启动前失败并说明原因，其他用例继续执行。
这些数字是软件支持范围，不代表远端精度测试已经通过。

原始 manifest 的 `golden=not_implemented` 是生成时的静态声明；当前批量运行器在
run_record 中启用 `semantic_validation`，运行后 measurement 的 `golden` 才是实际校验结果。
直接绕过批量运行器调用 `run_test.py` 不会自动应用这套语义输入。

## 先跑十个小规模实验

运行器默认按优先级排序，而不是源码目录或 manifest 的排列顺序：

1. 常用计算指令和 EXU2 候选的常见形式先跑，优先 FP32/FP16/BF16、32 位整数。
2. 同一优先级中先覆盖各指令的基础 II，再运行 forwarding，最后运行 16 流变体。
   因此中断前可以覆盖更多指令，并从同一批日志提取 latency。
3. 其余常见形式、较少使用的整数/谓词形式、FP8/打包形式依次排后。
   已知缺少语义输入编码的用例放在最后。

常用顺序是工程上的测试优先级，并非统计得到的指令频率。EXU2 候选来自仓内
`exu2_inventory.json` 和补充清单的 native-stage 项；实际路由仍以模型 dump 为准。
优先级规则在 `benchmarks/a6/priority.py`，常用指令顺序可调整 `COMMON_OPS`。

`--list` 与实际执行使用相同排序，`--order manifest` 可恢复原顺序。
结果根目录的 `execution_order.json` 保存本次选择的候选顺序；`--resume` 在其中跳过
已完成的匹配记录，不因排序变化重新跑已完成用例。排序只作用于 `--cases` 选择的套件，
不会自动把其他目录的指令加入任务。

先配置已验证的双包环境：

```bash
export CANN_HOME=/absolute/path/to/compiler/toolkit
export FULL_SIMULATOR_HOME=/absolute/path/to/debug/tools/simulator
source benchmarks/a6/set_env.sh

python3 run_a6_bench.py --cases op_kernel/a6_calibration --list
python3 run_a6_bench.py --cases op_kernel/a6_calibration \
  --repeats 3 --keep-lsu-dumps --output results/a6_calibration
```

`op_kernel/a6_calibration` 默认包含：

- VADD FP32：8/16 独立流 II、Vn/Vm 两种 forwarding，共四例。
- VAXPY FP32：8/16 独立流 II、Vn/旧 Vd 两种 forwarding，共四例。
- VLDS→VADD、VADD→VSTS：各一例。常量的另一输入由 Vdup 提供，不依赖另一条数据 load。

每个 forwarding 的依赖路径仍需检查实际 dump：编译器可能插入 VMOV、改变寄存器分配
或删除调用。目标数量不符或未找到实际 RAW 时，不接受该实验的时间参数。
两条跨单元探针是同一条 load/add/store 序列中的不同观察窗口，不声称另一源已由 ISU 证明就绪。

## 输入和校验输出

批量运行器自动生成 `_work/<case>/rN/inputs/input_B.bin`、`input_C.bin` 和
`input_profile.json`。`--input-seed N` 改变确定性的输入序列；同一 seed 的重复实验输入相同。
浮点基础输入使用正常非零数，支持 FP32、FP16、BF16 以及三种 FP8 格式；整数输入保持在小范围。
这不是随机边界值/特殊值测试集，NaN、Inf、subnormal 等应作为独立数据类别测量。

runner 通过批量脚本传入的 `MB_INPUT_DIR` 读取二进制文件，拒绝缺失或尺寸不匹配的输入。
通过 `MB_GUARD_ENABLE=1` 启用 GM canary，实际传给 kernel 的地址指向 canary 后的 tensor。
模型返回后检查 canary 和只读输入，通过时生成 `run/guard_check.txt`。
分析器还会将 `run/input_*.bin` 与生成的输入哈希核对，防止环境变量没有生效。

常见参考实现逐个检查每个 512 字节 tile 中前 256 字节的有效输出，覆盖两轮及所有中间结果。
tile 中其余字节目前未定义，不比较整块 tensor，也不能将它们解释为有效输出。
校验失败会阻止该实验被接受为成功观测，但校验成功也不自动成为硬件校准结果。

## 日志与结果解释

默认仍只归档原有六类 core0.veccore0 dump。`--keep-lsu-dumps` 额外保留该 core 的
LSU 日志，其他无关 dump 继续清理；不必为了 LSU 使用 `--keep-all-dumps`。

`measurement_semantics` 明确标出：

```text
latency:    done_minus_popped
ii:         same_exu_issue_gap
forwarding: dependent_popped_gap_not_proven_bypass
memory:     named_lsu_request_gap
```

`forwarding_by_operand` 分别汇总 Vn、Vm、old_Vd、store_data 等路径。多源同时依赖
保留为组合角色，不合并成一个已隔离的路径。原 `forwarding_min` 仅为兼容保留的整体最小观察值。

`summary.csv` 的 `observed_exus` 列显示本次目标指令实际出现的 EXU，例如
`EXU0|EXU2`；`summary.json` 同名字段保留数值数组 `[0, 2]`。
`exu_evidence=unavailable` 表示没有匹配证据，不能解释为 EXU0 或不能在 EXU 执行。
该集合来自各 stream 的 `observed_target_exus`，不是完整的硬件支持集合；例如只观察到
EXU2，并不能证明该指令只能在 EXU2 执行。复合 API/跨指令探针也不能据集合反推各阶段路由。
已有结果可执行 `python3 run_a6_bench.py --analyze-only --output /path/to/results`
重新汇总，不需要重跑模型；必须保留 run_record 和相应日志。

`convergence.json` 对每例的有效重复结果给出样本数、最小值、最大值。
至少三个重复实验全部产生同一个最小观察值时，`stable_observed_minimum=true`。
这只是重复性检查，**不是 8/16 流饱和性证明**；不同宽度的 summary 需另行对照，
更不能用同一 seed 的稳定结果排除数据相关时延。

## 生成其他变体和本地检查

```bash
python3 -m benchmarks.a6.variants --output op_kernel/a6_calibration \
  --ops VADD VAXPY --forms fp32 --ii-widths 8 16 --with-memory-pairs

# 只生成有明确模板支持的计算指令；不改变基础 787 个用例。
python3 -m benchmarks.a6.variants --output op_kernel/local_vsub_variants \
  --ops VSUB --forms fp32 --ii-widths 8 16

python3 -m unittest benchmarks.a6.test_suite benchmarks.a6.test_exu2 \
  benchmarks.a6.test_coverage_a5 benchmarks.a6.test_validation -v
```

本次 70 项回归测试通过，十个校准源码均通过本机 `dav-920r1-vec` 编译/链接检查；host runner 通过 CANN 头文件
下的 C++ 编译语法检查。本机没有可用的 A6 模型，尚未端到端验证 GM canary、数值参考
和新增探针在远端日志中的表现。先跑上述小集合，再扩展任务。
