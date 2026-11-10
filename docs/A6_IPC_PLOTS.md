# 批量画 CAmodel IPC

`plot_a6_ipc.py` 只依赖 Python 标准库，无需 matplotlib 或联网安装。它读取已有日志，不重新运行模型，不修改/删除原始结果。

在仓库根目录执行：

```bash
# 报告算子：每例画图，并叠加 original/software
python plot_a6_ipc.py --input results/a6_spill_run1 --output results/a6_spill_ipc --compare

# softmax：按 shape 叠加 no_membar/three_loop
python plot_a6_ipc.py --input results/a6_softmax_run1 --output results/a6_softmax_ipc --compare

# 一整个单指令批跑目录也可以；不要求存在 manifest/summary
python plot_a6_ipc.py --input results/a6_exu2_run1 --output results/a6_exu2_ipc
```

输出目录必须尚不存在。完成后用浏览器打开输出目录下的 `index.html`，可浏览全部图。

## IPC 口径

默认 `--metric done --scope compute --window 10`：

- 从 `EXU.dump` 找实际进入 EXU 的 `RV_V*` 向量指令，包含转换、广播、移动等，不只是浮点加乘。
- 用动态 ID + opcode 匹配 `instr_log.dump` 中的完成事件，按完成 cycle 计数。
- 不统计未在 EXU 日志出现的 LOAD/STORE、标量、控制指令；不统计谓词 `RV_P*` 指令。
- 不把每条向量指令展开成 lane 数；不是 FLOP/cycle，也不是 SIMT warp IPC。

每周期事件数是 `count[t]`；滑动窗口为：

```text
IPC[t] = sum(count[t-window+1 : t+1]) / window
```

窗口左端不足时按零填充，分母仍为 window。`--window 1` 就是逐周期计数。CSV 保留所有中间零事件周期，不只输出有事件的周期。

平均值 `average_ipc = 指令数 / (最后事件cycle - 首事件cycle + 1)`，这是所选事件跨度上的观测平均值，**不是整个 kernel 时间的 IPC**。图的默认零点是每条曲线自己的首个所选事件，不是 VF 启动时间。`--absolute-cycle` 可保留日志绝对 cycle，但不同运行的绝对起点不一定可比。

其他口径：

```bash
# EXU 发射 IPC，不是 popped IPC
python plot_a6_ipc.py --input results/a6_spill_run1 --output results/ipc_issue --metric exu-issue --window 1

# popped 边界，不代表 EXU 真正开始执行
python plot_a6_ipc.py --input results/a6_spill_run1 --output results/ipc_popped --metric popped

# 包含 EXU 上执行的谓词指令
python plot_a6_ipc.py --input results/a6_spill_run1 --output results/ipc_exu_all --scope exu

# 没有 EXU 日志时，显式改为统计全部可解析 RV 指令，口径不同
python plot_a6_ipc.py --input /path/to/log_ca --output results/ipc_all_rv --scope all-rv
```

`all-rv` 只包含现有解析器识别的、具有 cycle/ID/PC/RV opcode 的行，不包含 host/scalar 全部指令。不能与 `exu-issue` 合用。控制日志中同 ID 多阶段事件无法无歧义区分时，会拒绝生成该组图，而非猜测计数。

## 输出

- 每例一个 SVG，可直接插入支持 SVG 的文档或浏览器打开。
- 每例一个 CSV：`absolute_cycle, relative_cycle, instruction_count, window_ipc`。
- `summary.json`：指令数、事件跨度、平均 IPC、单周期峰值、窗口峰值、opcode 分布、观察到的 EXU、源日志列表及缺失情况。
- `index.html`：全部图的浏览入口。
- 使用 `--compare` 时额外生成叠加图。

只扫描 core0.veccore0，支持原始日志名、批跑归档前缀和 `.dump.1` / `.1.dump` 轮转命名。同目录、同前缀的轮转文件合并；完全相同事件去重。单条指令 ID/op 对应多个 cycle 时拒绝统计，避免把重复 launch/多段执行合成一条。多个独立运行目录不会合并计数。

叠加图要求归档名称以 `_original/_software` 或 `_no_membar/_three_loop` 结尾，且同组恰好两个不同版本。重复运行同名 case 时跳过叠加，单例图仍保留。比较使用统一坐标尺度；默认各自首事件归零。

缺日志、空日志或格式不识别会记录为 `error`；缺少部分 EXU 对应完成/popped 事件会记录为 `partial` 并保留图。它们都会让脚本最终退出码为非零，但不阻止其他用例出图。`partial` 的统计只代表实际匹配到的事件，不应用于完整性能结论。观察到的 EXU 集合也不等于硬件可执行单元全集。

默认拒绝跨度超过 2,000,000 cycles 的单例，可通过 `--max-cycles` 调整。SVG 超过 3000 cycles 时用每段最小/最大点保留包络，以控制文件大小；CSV 和 summary 始终使用完整逐周期数据。

脚本测试包括滑窗、零周期、重复轮转、缺失事件、ID 歧义、批跑分组和 SVG/XML 输出。另用已有 A5 CAmodel 日志做格式冒烟测试；这不代表已验证所有 A6 debug 包日志格式。A6 输出变更时请提供对应日志样例，不能仅凭空图判断 IPC 为零。
