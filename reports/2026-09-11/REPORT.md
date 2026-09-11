# JAG-Tree：RTX 5090 真实 7B pilot 结果（2026-09-11）

已从主分支 `be075ba` 拉取代码，修复 pilot CLI 接线，完成两轮模型生成、真实测试、逐边 LoRA 梯度和六组冻结回放。两轮运行均成功，候选库与结果校验通过。**科学状态仍为 INCOMPLETE：这不是正式 E1 或在线训练结果。**

192-token 屏幕配置中，JAG 的固定候选库精确梯度 MSE 比 uniform 高 16.85%；1024-token 配置中低 32.18%。两轮均只有 6/16 个任务出现至少一个全测试通过的候选。结果不足以支持稳定的 JAG 优越性，也未满足正式训练的前置门槛。

## 执行配置

- 硬件：RTX 5090，32607 MiB 显存；Python 3.12.14、Torch 2.14.0+cu130、Transformers 4.57.6、PEFT 0.20.0。
- 模型：`Qwen/Qwen2.5-Coder-7B-Instruct`，固定 revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`，BF16。
- 数据：BAAI/TACO 固定 revision `6e7429e9bcfb0e8d7aebfc719d98518f770b3985`，首个 train parquet 分片；按题面长度、原始行号排序选取 16 道 EASY Codeforces stdio 题，排除交互题和标注的构造算法题，保留完整测试。筛选在生成前完成。原机器 JSONL 不在仓库内，本轮是重新记录的新子集。
- evaluator：显式 `TrustedLocalSandbox`；降权运行、资源限制；不是外部认证的 OCI evaluator。
- predictor：`pilot:outcome-blind-structural-v1`，常量 value/gradient variance、零 cross covariance；LoRA rank 2/alpha 4，仅 q_proj，最后一个可训练 adapter tensor 的 256 维 sketch。不是正式 rank-32、多参数块梯度审计。
- 两轮采样 seed 17、projection seed 0；第二轮另显式固定 Torch 初始化 seed 17。第一轮上游 adapter 初始化未显式固定，封存候选库可精确回放，但从头再生成其梯度不保证逐位一致。

| 配置 | response cap | 分支深度 | 每分支子数 | nominal replay budget | 每组 draws |
|---|---:|---|---:|---:|---:|
| screen16 | 192 | 64/128 | 2 | 4096 | 21 |
| screen16-1024 | 1024 | 256/512/768 | 3 | 16384 | 37 |

两轮均为 4 roots。分支设置、预算和梯度初始化不同，不能将两轮差异单独归因于 response cap。第二轮只有 70 条边、68 个叶子，树大多在深分支前结束，因此也不能视为对深树优势的验证。

## 候选库结果

| 配置 | 真实梯度边数 | 候选数 | 全测试通过候选 | 至少一个通过候选的任务 | 触及长度上限 | unique generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| screen16 | 244 | 154 | 38 | 6/16 | 24 | 11495 |
| screen16-1024 | 70 | 68 | 18 | 6/16 | 0 | 8041 |

候选数随 EOS 提前结束而不同。上述候选通过比例不是 benchmark pass@1；每个方法共享同一个已封存候选库，并未各自重新生成答案。

## 六组比较

下面是固定候选库条件下、该组 draw 数的精确梯度均方误差相对 uniform 的比值，越低越好。不是模型测试准确率，也不是训练收益。

| arm | 192-token 配置 | 1024-token 配置 |
|---|---:|---:|
| uniform | 1.000000 | 1.000000 |
| entropy | 10.011526 | 1.391026 |
| value_variance | 1.000000 | 1.000000 |
| gradient_only | 1.000000 | 1.000000 |
| jag_full | 1.168481 | 0.678182 |
| oracle_moments | 0.203607 | 0.081015 |

uniform/value_variance/gradient_only 因常量 predictor 而重合。oracle 使用封存 reward，只是全信息参照，不能当作可部署结果。

每个候选路径有目标概率 p_i、分配概率 q_i、reward r_i、路径梯度 sketch g_i；目标梯度 μ = Σ p_i r_i g_i。对 n 次独立带放回 draw、维度 d，精确 MSE = [Σ p_i² r_i² ||g_i||² / q_i − ||μ||²] / (d n)。分析脚本逐条核对原始回放 q_i，并为每组做 8192 次重复采样交叉检查；两轮六组均通过检查。

原始 result.json 的 gradient_mse 是一次回放的 Monte Carlo 估计，存在明显采样噪声。尤其 1024-token 的单次估计中 JAG 看似更差，但精确条件 MSE 更低；原始文件未改写，分析结果单独保存。

各组 draw 数相同、均在 nominal budget 内，但实际消耗不同，且未用满预算。不是严格等实际 token 消耗的比较；逐组实际 replay_sampling_tokens 见 JSON。

## 验证与限制

- 完成模型权重加载和真实 BF16 CUDA 运算；第一次启动因缺少 C compiler 失败，安装 GCC 后成功，失败日志已保留。
- `jag-tree verify` 和 `verify_bank` 均通过；封存 scores 全部有限。
- 三道失败题 664/A、320/A、1562/A 用独立正确解通过完整测试。原模型失败包括算法错误、死循环、长度截断；检查结果在 evaluator-crosscheck.json。
- 本轮是 16 个预筛 EASY 题、单 seed、常量 predictor、小参数块 sketch；未训练校准 predictor、未证明 cross-fit joint moments 优势、未执行 optimizer update。
- 正式 E0/E1、256 calibration + 256 audit、跨 seed、严格资源配平与正式验证仍未完成；32GB 显存不能替代这些科学前置条件。

## 复现与文件

仓库代码位于 `/root/JAG-Tree`，本地分支 `fix/local-pilot-execution`。真实运行文件分别位于 `runs/screen16/` 和 `runs/screen16-1024/`，含 candidate bank、result.json、rows.jsonl、校验和、运行日志、环境与数据来源。

```bash
# 在仓库根目录，已安装 .venv 依赖后
.venv/bin/python -m jag_tree verify runs/screen16/run-ec0a8000dae0-seed17
.venv/bin/python -m jag_tree verify runs/screen16-1024/run-bc48a5dc9836-seed17
OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/2026-09-11/analyze.py runs/screen16 192
OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/2026-09-11/analyze.py runs/screen16-1024 1024
```

完整模型运行命令保存在各自 `provenance/execution.json`。新输出目录必须与已封存目录不同。环境依赖版本保存在 environment.txt；准备新机器时还需 GCC。

本次修复增加显式 `--sandbox trusted-local` 和 `--allow-pilot-predictor`，正式配置拒绝这两个选项；启动脚本支持 `trusted-local`，另增加逐任务日志、可复现数据筛选脚本和 1024-token pilot 配置。

GitHub 复现包：[代码与完整候选库归档](../../results/2026-09-11/JAG-Tree-pilots-2026-09-11.tar.gz)；SHA-256 `a2ca7c6a43930e9724653782c109443cd1fce22a7fc4ac77923d44d35e23c4ad`。
