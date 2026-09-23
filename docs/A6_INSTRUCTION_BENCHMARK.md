# A6 SIMD 单指令标定

## 1. 范围与当前验证状态

目标：为 A6 `Ascend910_9691 / dav_9201 / dav-920r1-vec` 采集单指令
latency、同指令 II、同指令 RAW forwarding。不是 SIMT 测试，也不以整个 kernel
运行时间替代指令周期。

测试设计参考 `VfSimulator/ascend_runner/operator_test.md`；覆盖原始
`VfSimulator/configs/isa.json` 的 **42 个条目、74 种 form**。
仓内 `benchmarks/a6/isa_inventory.json` 仅保留名字与 form，**没有复制 A5 周期参数**。
另补 6 类 A6 原生规约阶段，各有 fp32/fp16，以及 scalar VDUPS 的 3 种 form，共 15 个额外 form。
默认生成 **318 个 CCE 文件**，每个 kernel 只有一个被测 VF。

本机验证：318 个用例均通过 CCEC 的 `dav-920r1-vec` 编译与链接检查。
验证所用编译器：CANN 9.0.0 内的 clang 15.0.5，构建标识 2026-04-25；
另有 Python 标准库单元测试覆盖测量、双包传参、日志筛选和归档后重新解析。
源码 hash 与编译报告逐项核对一致。
本机没有可运行的 A6 simulator，**这些新用例尚未完成 A6 仿真验证，尚无 A6 标定数值**。
远程 CANN 版本仍需重新编译。搬运模板来自已经在用户服务器上通过精度验证的
`a6_scatter_probe.cce`，不代表所有新算子的正确性已被验证。

## 2. 远程执行

只需 Python 3.10+ 标准库，无需安装 numpy/pytest，也不依赖 VfSimulator 仓库。
以下命令在本仓根目录执行。替换两处安装路径，不要复制本机路径。

```bash
export CANN_HOME=/absolute/path/to/compiler-cann-toolkit
# 第二套包：只提供 simulator，不要 source 第二套包的 set_env.sh。
export FULL_SIMULATOR_HOME=/absolute/path/to/debug-package/tools/simulator
unset CCEC_EXTRA_FLAGS
source benchmarks/a6/set_env.sh
```

`FULL_SIMULATOR_HOME` 是包含 `dav_9201` 或 `Ascend910_9691` 子目录的目录，
其用法与本仓原有 `run_test.py` 相同。建议新 shell 只加载一套 CANN 环境；该脚本
不会删除你 shell 中已有的其他 CANN 路径。检查每次 `debug.log` 中实际解析的动态库。
编译器、头文件及 toolkit 来自 `CANN_HOME`，运行模型时优先加载
`FULL_SIMULATOR_HOME` 下所选 SoC 的 simulator 库。两套包仍须版本兼容。
若不设置 `FULL_SIMULATOR_HOME`，沿用 `CANN_HOME` 自带的 simulator。
批量脚本也支持 `--full-simulator-home /path/to/tools/simulator`，会校验目录结构。

先运行最小的 VADD，不要一开始跑全套：

```bash
python3 run_a6_bench.py --ops VADD --forms fp32 --output results/a6_vadd
```

运行 4 类源码：latency、II 宽度 8、II 宽度 16、forwarding；默认每类运行两次，
共 8 次独立 camodel 进程。设备为 0，block_dim 为 1。脚本显式固定 A6 参数，
不会沿用 shell 中旧的 A5 `CORE_ARCH`。

先看 `summary.csv`：VADD 的 `golden` 应为 `pass`，需要的周期列应有值。
若出现 `invalid_logs`，先查看对应 `measurement.json`，不要把空格理解成 0 cycle。

确认 VADD 日志可解析后，再执行全套：

```bash
python3 run_a6_bench.py --output results/a6_full --timeout 300
```

共 636 次模型启动，串行运行，每个用例独立进程、输出目录，默认单次超时 300 秒。
失败后继续后续用例。超时会终止该用例进程组，保留已有输出，返回码记录为 124。
默认输出根目录是 `results/a6_<时间戳>`。每次模型退出后，仅归档 veccore0 的
`instr_popped_log.dump`、`instr_log.dump`、`rvec.EXU.dump`、`rvec.IDU.dump`、
`rvec.ISU.dump`、`rvec.OOO.dump`；删除该次新建运行目录内的其他 dump。
不处理旧实验，不删除编译产物、输入输出数据、配置文件或 debug/model 日志。
这是运行后筛选，不会修改闭源 simulator 的日志开关，因此仍需预留单次运行的峰值空间。

常用命令：

```bash
# 只看清单，不编译运行
python3 run_a6_bench.py --list
# 只测选定指令及数据类型
python3 run_a6_bench.py --ops VADD VMUL VEXP VDIV --forms fp32 --repeats 3
# 只测依赖链
python3 run_a6_bench.py --ops VADD --modes forwarding
# 只编译 kernel 和链接二进制，不启动模型、不编译 host runner
python3 run_a6_bench.py --compile-only --output results/a6_compile
# 本机/服务器上的并行编译预检查；仅编译，并非并行仿真
python3 -m benchmarks.a6.check_compile --output results/a6_compile_parallel --jobs 4
# 续跑命令成功的相同实验；命令失败的保留旧目录并产生新 attempt
python3 run_a6_bench.py --output results/a6_full --resume
# 更新解析器后重新解析，完全不编译、不运行模型
python3 run_a6_bench.py --output results/a6_full --analyze-only
# 特殊排查时保留其余原始 dump（存放在 _work），例如需要 LSU
python3 run_a6_bench.py --ops VLDS --keep-all-dumps --output results/a6_load_debug
```

`--resume` 检查源码与环境指纹，不会覆盖旧结果。它跳过的是命令退出码为 0 的运行，
而不是声称精度/标定已通过；修改编译包内容、修改测试源码或希望重跑这些用例时使用新
`--output`。指纹不能识别安装路径不变而包内二进制被替换的情况。

## 3. 三种测量如何区分

### 3.1 latency

最小结构：`VLD -> 1 条被测指令 -> VST`。二元算术使用不同的 B/C 输入，
减少同源操作被代数化简的风险；这与手册同源单 load 模板的差别不影响同 ID
开始/完成配对，但本套不据此测手册中的单 load startup_cost。

```cpp
vlds(a0, ubB, 0, NORM);
vlds(b0, ubC, 0, NORM);
vadd(d0, a0, b0, m);
vsts(d0, ubA, 0, NORM_B32, out_mask);
```

严格使用同 core、veccore、动态 ID、PC、opcode 配对：

```text
start = instr_popped_log.dump 中的 cycle
done  = instr_log.dump 中的 cycle
latency = done - start
```

不额外加 1，不用 EXU 的 retire 字段代替 instr_log 完成点。得到的是手册约定的
观测 latency，不能自动认定它等于架构手册定义的内部功能单元延迟。

### 3.2 II(op, op)

先加载 8 或 16 组不同地址的输入；直线代码中计算独立结果，最后将每个结果写出。
不使用循环，不混入 loop 控制开销。不同输入地址防止等价表达式被合并，
结果全部保存防止死代码消除。编译器仍可能重排或折叠，因此必须检查 dump。

```cpp
vadd(d0, a0, b0, m);
vadd(d1, a1, b1, m);
// ... independent operands ...
vsts(d0, ubA, 0, NORM_B32, out_mask);
vsts(d1, ubA, 64, NORM_B32, out_mask);
```

EXU 日志按 `exu_id` 分组，仅比较同 EXU 的相邻发射事件，且两个事件都必须
对应实际 popped 中的被测指令。排除存在 RAW/WAR/WAW 的寄存器对。
**不存在跨 EXU 或全局 popped 间隔作为 II 的 fallback。**

输出 `ii_min` 是独立流的最小观测间隔。若等待 load、前端带宽、EXQ 或寄存器
压力成为瓶颈，该间隔仍会偏大。因此必须比较 w8/w16、重复运行及原始事件，
收敛且没有资源阻塞后才将其作为硬件 II 候选；脚本不自动回填模型参数。

### 3.3 forwarding(op, op)

使用两条同指令真实数据依赖，不用循环迭代间隔代替 forwarding：

```cpp
vadd(d0, a0, b0, m);
vadd(d1, d0, b1, m);
// Both d0 and d1 are written out.
```

按实际汇编的 PC 顺序跟踪寄存器最后写者，确认 consumer 的源确实来自 producer，
再计算 `consumer_start - producer_start`。两个事件使用 popped 周期。
不同功能单元之间是否有差异需看原始 EXU 日志；这里的值是观测依赖间隔，仍可能
受到其他源未就绪或资源争用影响，不自动等同于最短硬件 bypass 延迟。

例如 VABS 连用可能被编译器化简为一次，目标指令数不足时会报 `invalid_logs`，
不会输出伪造 forwarding。若远程编译器发生这种优化，应检查实际汇编，后续另行
设计保留该依赖的汇编 probe，而不是通过插入其他计算去冒充 self forwarding。

## 4. 非普通算术的处理

| 类别 | 处理方式 |
| --- | --- |
| VCMAX/VCMIN/VCADD/VCGMAX | 保留 API 展开探针，标记 `compound_api`；不生成单指令 timing。额外测 VCGMAXV2/VCGMINV2/VCGADDV2/VCMAXV2/VCMINV2/VCADDV2 |
| VCVT、VMULSCVT | 测 latency/独立流 II；输入输出类型不同，不构造非法的同 form RAW 链 |
| VCI/VBR | 标量参数生成向量，无同指令向量 RAW 输入；self forwarding 标记不适用 |
| VDUP | 选择 vector-to-vector、POS_LOWEST 形式，可构造 RAW；另有 VDUPS 用例测 scalar 形式，不混合两种数据路径 |
| VCMP_EQ | 输出 predicate，借 VSEL 写出结果；本版不测 predicate 自 forwarding |
| VSHLS/VSHRS | isa 的 fp32/fp16 标签按 32/16 位整数位移实现；manifest 记录真实 uint 类型 |
| VPACK/b32 | 明确测 u32 到 u16 的 LOWER 位打包，不解释为 bf16 数值计算 |
| VLDS/VSTS | 测 load/store 本身的 start/done；没有向量寄存器 self RAW 链 |
| VSTUS/VSTAS | 每条流使用独立 align 状态，VSTUS 后跟 VSTAS 排空，分别选择对应 opcode 测量 |
| VSSTB | b16、小 mask、独立 UB tile 的布局探针，不代表所有 scatter/bank-conflict 形态 |

这里的“不适用”是指当前有正常数值语义的测试模板，不是在断言硬件没有这种
forwarding 通路。类型转换可以另外设计无额外指令的寄存器位重解释实验，但必须先
核对实际 ISA 的读写布局及特殊值行为，本版没有将这类实验冒充普通同类型依赖链。

内存指令不在 EXU 执行。独立流额外从 LSU 中提取明确的 `SEND_UB_RD.PORT_n`
或 `STU_IB_BUF.ISSUE` 事件，分别按端口/阶段统计 `memory_interval_min`。
这不是 EXU II，读请求间隔也不一定等于 load 指令入口 II；保留事件名供建模者确认。
若一条指令产生多个相同资源请求或日志无此阶段，不推算该数值。
**默认六文件筛选不保留 LSU，因此 `memory_interval_min` 留空。** 如需此项，
运行时加 `--keep-all-dumps`。普通计算指令的 latency、EXU II、RAW forwarding
所需的 popped/instr/EXU 都在默认保留范围内。

本机 CANN header 中 A6 的 API 拆分、签名是生成器依据；远程编译器实际产生的
opcode 以 dump 为准。未识别的名字保留在 histogram 中，不通过模糊匹配强行套入。

## 5. 输入输出与精度

CCE 的参数顺序固定为输出 A、输入 B、输入 C。typed tensor metadata 由原有
generic runner 读取，输入使用 runner 的确定性初始化，实际数据归档到 input_B/C.bin。
每个独立输入块是一个 256-byte SIMD 向量；本套未扫描不同 mask/向量长度。
VLN 特意使用 C 的正数输入（runner 初始化范围 1.69..2.65），保证两级 log 链
不因第一级产生负数而使第二级落入非法定义域。

UB 布局：B 在 0x0000、C 在 0x4000、A 在 0x8000、scatter offset 表在 0xc000。
默认最大 16 个独立向量，每个数据区小于 16 KiB。GM->UB 用 NDDMA，
MTE2->V 同步后执行 VF，再经 V/S->MTE3 同步，用 offset_type=0、offset=0 的
单 burst scatter 写回，最后等待 MTE3 完成。

解析阶段对 fp32/fp16 的常用算术、exp/log/sqrt、VLDS/VSTS 执行独立 golden：
读取保存的输入，逐元素计算，fp16 链每步舍入，检查所有输出，包括中间值。
fp32 容差为 `1e-4 * max(1, abs(expected))`，fp16 为 `5e-3 * max(1, abs(expected))`。
这是 timing probe 的功能冒烟检查，不是算子级 ULP 精度验收。

转换、规约、位操作和特殊布局等尚无 golden 的用例，明确记录
`golden=not_implemented`，**不能将其解释为精度通过**。部分转换或规约仅定义部分
lane 的数值，此类文件保存整块原始输出用于后续 ISA 语义核验，不能直接拿全块做逐元素 golden。
出现 golden fail 时不输出该用例的 timing，避免将未正确运行的程序用于标定。

## 6. 结果结构与失败判读

```text
results/a6_full/
  summary.json
  summary.csv
  VADD/
    vadd_fp32_latency/r0/
      vadd_fp32_latency__r0__core0.veccore0.instr_popped_log.dump
      vadd_fp32_latency__r0__core0.veccore0.instr_log.dump
      vadd_fp32_latency__r0__core0.veccore0.rvec.EXU.dump
      vadd_fp32_latency__r0__core0.veccore0.rvec.IDU.dump
      vadd_fp32_latency__r0__core0.veccore0.rvec.ISU.dump
      vadd_fp32_latency__r0__core0.veccore0.rvec.OOO.dump
    vadd_fp32_ii_w8/r0/         # 同样六类文件，前缀为对应 case
    vadd_fp32_ii_w16/r0/
    vadd_fp32_forwarding/r0/
  VEXP/                       # 其他指令同理
  _work/vadd_fp32_latency/r0/
    kernel.cce                 # 本次源码快照
    run_record.json            # 命令、环境、源码指纹、退出码、wall time
    console.log                # run_test.py 的进度输出
    debug.log                  # 编译和环境诊断
    model.log                  # 原始模型/runner 输出
    build/                     # .o、链接产物、runner
    run/
      input_B.bin
      input_C.bin
      output_A.bin
      log_ca/                  # 六类 dump 已移到 VADD 下，其余默认清理
    measurement.json           # 指令直方图、周期样本、core、ID、寄存器、源文件及行号
```

文件名保留指令、form、测试类型、II 宽度、repeat、core/veccore 标识，因此不同
用例不会互相覆盖。仅选择 veccore0，但不同 core 的 veccore0 不混并。
归档是移动而非复制，不额外保留一份相同 dump；轮转文件也保留原始编号。
缺失或为空的六类文件会给出 WARN，并记录在 `run_record.json` 的 `dump_archive` 中。
不会创建假的空文件来凑齐六份。归档记录使用相对路径，复制整个结果目录后仍可重新解析。
当前周期解析器使用未轮转的主文件；发生轮转时保留分卷供检查，不将不完整样本视为全量。

`status=compiled` 只代表预编译通过；`observed` 代表有所需周期样本，不代表已标定。
`failed` 是编译/运行/数值失败，`missing_logs` 是缺少 popped 日志，
`invalid_logs` 是无法从现有日志确定该指标，`compound_api` 是兼容 API 展开。
`not_collected` 表示当前筛选策略未保留该指标所需日志（例如 LSU），不代表模型运行失败。
null/CSV 空格均表示未知，不是 0。summary 不会改写原来的 isa/II/forwarding 配置。

全量日志生成取决于 simulator 包及其配置，脚本无法让 release 包输出不存在的 EXU 日志。
只有 stars 日志、或只有 popped/instr 而没有 EXU 时，不要开展全套 II 标定。

## 7. 扩展与重新生成

仓库已包含全部 .cce，可以直接运行，不必在远程机器访问原始 E 盘文件。
修改模板在 `benchmarks/a6/generate.py`，修改完成后重新生成：

```bash
python3 -m benchmarks.a6.generate
# 使用更新后的 ISA 清单；不把其中 timing 当成 A6 真值
python3 -m benchmarks.a6.generate --isa /path/to/updated/isa.json
# 改独立流宽度，建议输出到新目录，避免混杂旧源码
python3 -m benchmarks.a6.generate --widths 4 8 16 32 --output op_kernel/a6_width_sweep
python3 run_a6_bench.py --cases op_kernel/a6_width_sweep --ops VADD --forms fp32
python3 -m unittest benchmarks.a6.test_suite -v
```

manifest 记录每个 form 的覆盖、目标 opcode、预期动态数量、源码 hash、特殊说明。
新增未知指令需要补充明确的调用签名和语义，不能仅在清单中加名字便假定生成器懂得其含义。
日志格式适配在 `benchmarks/a6/analyze.py`，改动后应运行单元测试，再对旧结果
`--analyze-only`。返回结果时优先提供 VADD 的两个 II 宽度和 forwarding 的完整目录，
这样能核对 A6 实际的 EXU 事件格式、优化结果和周期口径。
