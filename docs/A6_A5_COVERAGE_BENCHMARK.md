# A6 指令覆盖补充：对照 A5 清单

后续输入与校准检查见 [A6_CALIBRATION_VALIDATION.md](A6_CALIBRATION_VALIDATION.md)：
批量运行已增加语义 dtype 输入、部分数值参考和 GM 保护检查。原始源码清单不变，
运行结果明确标为尚未校准的观测值；不能仅凭编译成功直接采纳时序参数。

## 1. 范围与目录

本套件对照本地 Git 跟踪分支 `tilesim/wch/VfSimulator-refactor` 的
`configs/isa.json`，固定来源提交为 `1203846`。未在线更新该分支。
快照保存在 `benchmarks/a6/a5_coverage_inventory.json`，只提取指令及 form，
**不复制 A5 的 latency、II、forwarding 或执行单元路由参数**。

清单共有 981 个“指令 + form”组合。现有 `a6_instruction` 与 `a6_exu2`
已提供其中 229 项的源码模板，本次针对剩余 752 项补充，原有两套用例不变。
“已有模板”不代表已经完成远端仿真或时序校准。

本次生成 **787 个 CCE 用例：589 个 II、198 个 forwarding**，覆盖缺口中的
589 个指令/form。剩余 163 项明确列为 deferred，不加入默认运行列表。
按源码模板计，参考清单中共有 818/981 项被三套测试覆盖；这不是实测覆盖率。

```text
benchmarks/a6/coverage_a5.py               生成器
benchmarks/a6/a5_coverage_inventory.json   固定的 A5 指令/form 清单
benchmarks/a6/a5_coverage_deferred.json    已确认的编译器后端拒绝项
benchmarks/a6/raw_wrapper.py              A6 NDDMA + scatter 搬运外壳
benchmarks/a6/test_coverage_a5.py           结构与日志分析回归测试
op_kernel/a6_a5_coverage/*.cce             可独立编译的新增用例
op_kernel/a6_a5_coverage/manifest.json      用例、源码哈希和逐项覆盖状态
```

主要补充比较与谓词、整数/位运算、移位、累加、进位运算、宽乘法、类型转换，
以及普通/非对齐访存、gather/scatter 等缺失的类型和寻址形式。
每个用例都带 GM 输入、UB 工作区、GM 输出，不是 VF-only 片段。

`manifest.json` 的 `coverage` 是完整的覆盖依据：

| 状态 | 含义 |
| --- | --- |
| `existing_probe` | 前两套已有相同指令/form，本套不重复生成 |
| `new_probe` | 本套生成了 CCE 文件，具体文件见 `cases` |
| `deferred` | 尚未可靠适配，`reason` 记录原因，不进入默认批量任务 |

64 位多寄存器、部分打包格式、尚未核实的 A6 签名等保留为 deferred。
本机编译器拒绝或崩溃也不等同于 A6 硬件不支持。

## 2. 远端运行

沿用已跑通 VADD 的两个 CANN 包，**只 source 编译包的环境**。
由项目脚本将 simulator 路径切换到完整日志包，不额外 source 第二个包。

```bash
cd /absolute/path/to/microbenchmark-cce-direct
export CANN_HOME=/absolute/path/to/compiler/toolkit
export FULL_SIMULATOR_HOME=/absolute/path/to/debug/tools/simulator
source benchmarks/a6/set_env.sh

# 只列出新增任务，不编译、不启动模型。
python3 run_a6_bench.py --cases op_kernel/a6_a5_coverage --list

# 先跑少量用例，检查目标指令是否实际存在、是否保留了 RAW 链。
python3 run_a6_bench.py --cases op_kernel/a6_a5_coverage \
  --ops VADDC VMULL --forms int32 --output results/a6_a5_smoke

# 全量跑本次补充，不会重复跑前两套。
python3 run_a6_bench.py --cases op_kernel/a6_a5_coverage \
  --output results/a6_a5_coverage

# 上一个进程退出后续跑。
python3 run_a6_bench.py --cases op_kernel/a6_a5_coverage \
  --output results/a6_a5_coverage --resume
```

默认 device 0、block_dim 1、每个用例运行一次，顺序执行且无超时限制。
可用 `--timeout 7200` 设置每例“编译 + 仿真”的两小时上限。
目标为 `Ascend910_9691` / `dav-920r1-vec` / `dav_9201`；不要改成 A5 目标来编译。

默认已启用常用指令/EXU2 候选优先排序，常见类型的基础 II 先跑，再覆盖 forwarding
和少见形式。`--list` 即实际候选执行顺序，输出中另存 `execution_order.json`。
不改变源码、用例名称或 `--resume` 指纹；需要旧顺序时加 `--order manifest`。

## 3. 输出与日志

```text
results/a6_a5_coverage/summary.csv
results/a6_a5_coverage/summary.json
results/a6_a5_coverage/<OP>/<case>/r0/<case>__r0__core0.veccore0.*.dump
results/a6_a5_coverage/_work/<case>/r0/debug.log
results/a6_a5_coverage/_work/<case>/r0/model.log
results/a6_a5_coverage/_work/<case>/r0/measurement.json
```

默认只归档 core0.veccore0 的 `instr_popped_log.dump`、`instr_log.dump`、
`EXU.dump`、`IDU.dump`、`ISU.dump`、`OOO.dump`，文件名带指令和用例名称。
其他 dump 在该次模型正常退出或报错返回后清理，编译与诊断文件保留。

**访存 II 不能从 EXU 发射间隔推断。** 本套给访存用例设置 `memory_probe`，
默认六类日志不含 LSU，此时仍可提取匹配指令的起止间隔，但访存发射间隔标为
`not_collected` 或缺少证据。确实需要访存间隔时，为单独的小批任务加
`--keep-lsu-dumps`，分析器从支持的 LSU request/issue 事件提取间隔。
这只额外保留 LSU；若新模型 LSU 格式不同，则仍需适配解析器。

## 4. 测试结构与解释边界

- II 模板在循环内放 8 个独立目标调用；普通模板执行两轮。
  VCI/VBR/VDUPS 等常量源模板只执行一轮，避免跨循环常量提升。
- Forwarding 模板使用 A0,A1,B0,B1,C0,C1：A0→B0→C0 与 A1→B1→C1
  两条真实寄存器依赖链。中间结果也写出，避免只保留最后一级。
- Latency 从同一批日志中按指令 ID、PC、opcode 配对起止事件，不另造 latency 用例。
- 比较输出、进位输出和宽乘法高半结果均保留。Merging 和累加型目标寄存器先初始化。
- 类型变化、纯输出、访存等不适合本模板自身 RAW 链的项目只提供 II/观察用例，
  不伪造 self-forwarding。非对齐访存包含 alignment 初始化/flush 辅助指令，
  测得的是该序列下的观察间隔，不保证已饱和得到硬件极限 II。
- 新模板通过 uint32 传输原始位，再按目标向量类型解释，避免输入阶段隐式插入 VCVT。
  类型转换用例中的 VCVT 才是待测目标。各测试块留有输出间隔，部分无效 lane 和
  填充区域未定义，不能直接把整个输出张量视为数值结果。
- 原始 manifest 的 raw-bit 用例标记为 `golden=not_implemented`。当前批量运行器
  已增加 245 个用例的数值参考，实际是否通过以 measurement 为准；其他形式仍明确
  不提供数值校验。正式校准前还应检查实际目标指令、依赖链和执行单元。
- A5 API 名称不一定等于 A6 指令名。例如 A6 VMOV_MERGE 的头文件实现为 VSEL；
  `target_aliases` 记录已知映射。编译器删指令、重排、复合展开或新日志格式导致
  目标计数不符时，分析器报告无效/待核查，不当作零延迟或已测得 II。

## 5. 再生成与验证

787 个最终源码 SHA256 均已对应到成功的本机编译/链接记录，目标为
`dav-920r1-vec`，编译器为 CANN 9.0.0 中的 clang 15.0.5（2026-04-25 构建）。
回归测试共 55 项。PLDI/PSTI 的 b8 模板被该编译器后端以
`Cannot select intrinsic llvm.hivm.pldi.b8 / psti.b8` 拒绝，已排除；
远端新编译器是否支持应独立验证，不能据此判断硬件不支持。

远端不需要 VfSimulator checkout，默认使用仓内快照：

```bash
python3 -m benchmarks.a6.coverage_a5
python3 -m unittest benchmarks.a6.test_suite benchmarks.a6.test_exu2 \
  benchmarks.a6.test_coverage_a5 -v

# 只编译检查，不启动模型。output 使用新的目录。
python3 -m benchmarks.a6.check_compile --cases op_kernel/a6_a5_coverage \
  --cann-home "$CANN_HOME" --output results/a5_coverage_compile_check --jobs 4
```

显式更新参考分支时才使用：

```bash
python3 -m benchmarks.a6.coverage_a5 \
  --source-repo /absolute/path/to/VfSimulator \
  --source-ref remote/wch/VfSimulator-refactor
```

生成器遇到被手工修改的既有生成文件会拒绝覆盖。探索性修改可先生成到新的
`--output` 目录。本机没有可运行的 A6 simulator，因此本次不提供实测时序结论；
请在远端使用匹配编译器和完整日志模型运行上述命令。
