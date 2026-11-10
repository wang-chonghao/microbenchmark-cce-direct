# A6 Membar 最小测量用例

目标是分开观察屏障发射、同步放行、退休、后续访存开始及相邻屏障推进。
四个完整 CCE kernel 均使用 A6 DMA 包装，编译目标固定 `dav-920r1-vec` / `__NPU_ARCH__ == 9201`。
不改 VfSim，不把尚未观察到的 A6 时序填进配置。

## 四个用例

| 文件 / 目录名 | 被测部分 | VF 内 LOAD / STORE | 屏障总数 |
|---|---|---:|---:|
| `01_cluster_vld_vst` | 连续 8 条 VLD_VST | 1 / 8 | 8 |
| `02_cluster_vst_vld` | 连续 8 条 VST_VLD | 9 / 16 | 10 |
| `03_interleave_store` | 8 组 VLD_VST → STORE | 1 / 8 | 8 |
| `04_interleave_load` | 8 组 VST_VLD → LOAD | 9 / 16 | 10 |

02/04 的另外两条 VLD_VST 分别用于初始化 scratch、阻止输出 STORE 提前影响被测 LOAD；不计入目标 8 条。
全部手写展开，无 VLOOP、reduce、EXP/DIV 或算术链。最多 9 个向量变量和一个全有效谓词。
默认剔除目标屏障首尾各一条，用中间 6 条统计，同时保留全部逐条证据。

每条普通访存为 256B，CCE offset 以 FP32 元素计，因此 0/64/128/... 对应相邻 256B 块。
UB input 位于 `0x0000`，output 位于 `0x1000`，scratch 位于 `0x2000`，DMA offsets 位于 `0x3000`，互不重叠。
GM input 为 64 个 FP32；GM output 的前 512 个 FP32 为 8 份 input，末尾另外分配 256B guard。
`host.cpp` 逐字节验证八块输出及 guard，不采用近似数值阈值。

## 编译与运行

仅编译，不需要 A6 simulator：

```bash
python3 cce_code/a6_membar_timing_probe/build.py \
  --toolkit /path/to/ascend-toolkit \
  --out /tmp/a6_membar_probe_build
```

编译保持完整 DMA/VF，不删除外围代码；固定 `-cce-aicore-vec-misched=0`、VF fusion 关闭、VF stack 为 0。
任一编译失败就停止，不自动放大 stack 来容忍 spill。每个 case 保存源码/公共头/host 快照、编译/链接命令及哈希。
`--out` 必须为新目录或空目录，避免覆盖原始记录。可用 `--cases 03_interleave_store` 只编一个。

有实际匹配的 A6 CAModel 时：

```bash
python3 cce_code/a6_membar_timing_probe/build.py \
  --toolkit /path/to/ascend-toolkit \
  --out /tmp/a6_membar_probe_run \
  --run --npu-type ACTUAL_A6_MODEL_NAME
```

`ACTUAL_A6_MODEL_NAME` 必须替换为本机真实 A6 simulator 的目录名。脚本不会自动选 A5。
编译目标检查不能证明 simulator 型号；需自行确认 SDK 与模型版本。golden 正确也不等于已确认芯片代际。
host 默认 block_dim=1、device=0。可用 `--host-arch aarch64-linux` 指定 host SDK 目录。
工具覆盖变量为 `MEMBAR_CCEC`、`MEMBAR_LINKER`、`MEMBAR_HOST_CXX`；附加 host 路径可用
`MEMBAR_CXX_INCLUDE_DIRS` / `MEMBAR_EXTRA_LIBRARY_DIRS`（冒号分隔）。

若使用自己的运行框架：入口名称见 `experiments.json`，参数均为 `(GM float* input, GM float* output)`；只启动一个 block、每个 kernel 执行一次，并开启详细日志。

## 提取日志

每个用例的日志直接放在其 case 目录下，不混入其他核或其他运行的日志：

```text
logs/
  01_cluster_vld_vst/
    core0.veccore0.instr_popped_log.dump
    core0.veccore0.instr_log.dump
    core0.veccore0.rvec.ISU.dump        或包含 SU 事件的详细 .dump/.log
    core0.veccore0.rvec.simd.depmgr.dump
    core0.veccore0.rvec.LSU.dump
  02_cluster_vst_vld/...
  03_interleave_store/...
  04_interleave_load/...
```

支持 `.dump.gz`。文件名应含 `ISU`、`LSU`、`GSU`、`SU.` 或 `depmgr`；默认事件格式兼容仓内 A5 详细日志，并识别 SU/RVECSU/REVCSU 控制标签，**不是已确认所有 A6 CAModel 版本的日志格式**。

```bash
python3 cce_code/a6_membar_timing_probe/extract_membar_timing.py \
  --logs /tmp/a6_membar_probe_run --out /tmp/a6_membar_measurement
```

单独分析一个用例：

```bash
python3 cce_code/a6_membar_timing_probe/extract_membar_timing.py \
  --case 03_interleave_store --logs /path/to/that_case_logs \
  --out /tmp/a6_store_measurement --control-unit SU
```

多个 VF 时必须显式 `--vf-index`（从 0 开始）。一个 VF 内出现重复 PC/动态循环、屏障数量变化、额外搬运或 spill、非预期计算时拒绝提取。
普通日志中的 RVECLD/RVECST 两侧记录按 `(PC, ID)` 合为一条指令，但原阶段记录全部保留；不会取最早 popped 冒充控制发射时间。

如果 SU 详细日志的字段名不同，用 `--patterns event_patterns.json` 提供匹配规则。
`event_patterns.example.json` 只是模板，不代表真实 A6 日志已经实测为该格式。
每条规则必须捕获 `(?P<id>...)`；issue/retire/receive 还需捕获 `(?P<unit>...)`。
可选捕获 `cycle` 和十六进制 `pc`；否则从行首数字方括号和 PC 字段读取。
只覆盖提供的规则，其余继续使用默认；可覆盖键见脚本中的 `PATTERNS`。

## 如何读结果

- `overview.json`：按 VLD_VST/VST_VLD 汇总候选；`calibrated=false`，不自动写 uarch。
- 各 case 的 `barriers.csv`：8 条屏障的 I/G/R/C、下一条 I、时间差和重叠标记。
- 各 case 的 `summary.json`：原始两侧事件、详细事件文件及行号、完整直方图、输入计数与缺失项。
- 各 case 的 `README.md`：可直接阅读的统计表。

I 为详细控制发射，G 为 depmgr 消费侧放行，R 为同一控制阶段退休。
VLD_VST 的 G 使用 STU consumer dec，VST_VLD 的 G 使用 LDU consumer dec。
LSU 的 GO 事件另存为证据，不混成相同阶段；A6 如改变同步协议，应根据实际日志修改 release 规则。
STORE 的 C 对应当前模型的 SHQ ISU_ISSUE；LOAD 的 C 使用 LSU `SEND_UB_RD`，不采用过早的 LDQ popped。
如果 A6 阶段不同，需明确选择模型 start 的对应事件；不能混用 `SEND_UB_WR` 与 STORE 的 SHQ 发射。

连续屏障给出 `G-I`、`R-I`、`I_next-R`；发射前目标类别访存未完成的样本不用于 release/retire 候选。
交错用例给出 `C-G`，但仍需排除数据、端口和前端到达造成的阻塞。
候选至少需要 3 个相同差值，否则为 null，不取一次最小值冒充常数。

**两个重要限制：**

1. `I_next-C` 只列为反馈延迟的观测上界，可能还被前一屏障退休等条件限制，不能直接当成 `lsu_start_to_next_issue`。结合连续屏障的 `I_next-R` 判断哪一项占主导，再做针对性对照。
2. `admission_delay` 不独立可辨识，明确输出 null，不默认测成 1。若 `I_next < R`，脚本标记屏障重叠，正的 `next_issue_delay` 无法解释该现象，需调整控制器结构。

仅有两份 instr 日志时仍保留计数和原始事件，但详细时间缺失会显示 null；不会据 VF 总周期或 SU 标签推算内部阶段。
日志统计本身不证明 golden 通过；使用配套 host 运行可独立完成精度检查。

## 当前验证边界

2026-11-10：四个完整 kernel 在本机 CANN 9.0 beta.1 下，以 A6 目标和 VF stack=0 编译/链接通过，未删 DMA。
host 做语法检查；提取器通过合成日志测试，并用历史 A5 clustered8 原始日志交叉验证
VLD_VST 的 18/20/1、VST_VLD 的 7/9/2 三项差值。
当前本机 SDK 无对应 A6 simulator，**未声称 A6 golden、时序或屏障数量已运行验证**。

```bash
python3 -m unittest discover -s cce_code/a6_membar_timing_probe -p test_extract_membar_timing.py
```
