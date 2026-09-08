<img width="1280" height="824" alt="C10948B8F3A2DDE8CA19EA59F20B23B1" src="https://github.com/user-attachments/assets/d25abaed-2bbd-44fc-84e2-676eeb2cb628" /><img width="1280" height="824" alt="C10948B8F3A2DDE8CA19EA59F20B23B1" src="https://github.com/user-attachments/assets/fc3391e1-fc64-4a77-ab28-a3a585adb31c" /># Franka-Manipulation-RL

## 项目简介

本项目基于 **Isaac Lab + RSL-RL + PPO**，将此前四足机器人 Locomotion 强化学习经验进一步迁移到 **Franka Panda Manipulation**，完成：

**Reach → Grasp → Lift → Transport**

项目重点研究 **Reward Design 对机械臂操作学习过程和最终行为的影响**，并通过多随机种子训练和独立 Evaluation 分析不同奖励结构下的收敛特性与 Failure Mode。

---

## 项目演示

### Staged Reward：完整 Manipulation

机械臂完成完整的：

**Reach → Grasp → Lift → Transport**

<img width="1280" height="824" alt="C10948B8F3A2DDE8CA19EA59F20B23B1" src="https://github.com/user-attachments/assets/7eef76a2-b477-4d5b-a3a1-491e5ce43431" />

### Baseline：Transport Drop Failure

Baseline 策略能够完成 Lift，但部分策略会在后续 Transport 阶段发生物体掉落。

![alt text](9e5a60b2599d996d7b9fafd3692b152a.gif)
---

## 我做了什么

### 1. 搭建 Franka Manipulation 强化学习任务

基于 Isaac Lab 官方 Franka Lift Task 搭建独立项目，完成 **PPO 大规模并行训练流程**，并建立 Baseline。

### 2. 设计并迭代 Staged Reward

针对 Manipulation 的任务结构设计：

**Reach → Grasp → Lift → Transport**

阶段化 Reward，并与官方 Baseline Reward 进行对比。

训练过程中发现第一版 Grasp Reward 出现 **Reward Hacking**：策略会靠近方块并闭合夹爪获取奖励，却不真正完成抓取和抬升。

随后将其修改为基于左右 Fingertip 与方块空间关系的 **Grasp Alignment Reward**。

同时将 Lift 判断修改为相对于方块初始高度的：

\[
z-z_0>0.04m
\]

避免将方块初始高度误判为成功 Lift。

### 3. 建立独立 Evaluation Pipeline

编写独立评估程序，不直接使用 Training Reward 判断策略性能，而是统一评价：

- Lift Success Rate
- Goal Success Rate
- Drop Rate
- Final Position Error

Baseline 与 Staged Reward 分别训练 **3 个随机种子**，每个策略进行 **256 个 deterministic evaluation episodes**，共完成 **1536 个 Evaluation Episodes**。

---

## Experiment 1：Reward Design

| Metric | Baseline | Staged Reward |
|---|---:|---:|
| Lift Success ↑ | **98.57 ± 2.48%** | 85.16 ± 19.05% |
| Goal Success ↑ | 25.65 ± 19.91% | **30.73 ± 27.02%** |
| Drop Rate ↓ | 14.71 ± 25.48% | **0.52 ± 0.90%** |
| Final Position Error ↓ | 0.2847 ± 0.3453 m | **0.1881 ± 0.0996 m** |

### 主要结果

- **Baseline 学习 Lift 更快、更稳定**；
- Staged Reward 在固定训练预算下收敛更慢，优化难度更高；
- Staged Reward 明显降低了 **Object Drop Failure**；
- 两种 Reward 会形成不同的学习过程和 Failure Mode。

因此，Staged Reward 并不是简单提高所有指标，而是在：

**Task Structure 与 Optimization Difficulty**

之间形成权衡。

对 Staged Reward 的 Checkpoint Evaluation 进一步发现，其 Lift Success 从 Iteration 300 的 **0%** 持续提升至 Iteration 500 的 **89%**，继续训练后仍能明显提高，说明其主要问题之一是 **收敛速度与 Sample Efficiency**，而不是无法学习任务。

---

## Failure Analysis

实验中观察到两类典型失败：

**Baseline：**  
能够较快学会 Lift，但部分策略在 Transport 阶段频繁掉落物体。

**Staged Reward：**  
完整任务结构更加明确，但部分随机种子在固定训练预算内收敛较慢。

这说明仅观察 Lift Success 或 Training Reward，并不足以判断 Manipulation Policy 是否真正学会完整任务。

---

## 下一步：Position Generalization

后续计划研究：

**训练阶段的位置随机化是否能够提高策略对未见 Object / Target Position 的泛化能力。**

将比较 Narrow Training Distribution 与 Wider Randomized Training Distribution，并进行 ID / OOD Evaluation。

---

## 项目结构

```text
Franka-Manipulation-RL/
├── README.md
├── source/
│   └── franka_manipulation_rl/
│       └── franka_manipulation_rl/
│           └── tasks/
│               └── manager_based/
│                   └── franka_manipulation_rl/
│                       ├── franka_manipulation_rl_env_cfg.py
│                       ├── mdp/
│                       │   └── rewards.py
│                       └── agents/
│                           └── rsl_rl_ppo_cfg.py
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py
│   │   └── play.py
│   └── evaluation/
│       └── evaluate_policy.py
└── results/
    ├── evaluation/
    ├── figures/
    └── videos/
```

### 主要文件说明

- `franka_manipulation_rl_env_cfg.py`：定义 Baseline 与 Staged Reward 环境配置
- `mdp/rewards.py`：实现 Reach、Grasp Alignment、Relative Lift、Goal Tracking 等自定义奖励
- `agents/rsl_rl_ppo_cfg.py`：RSL-RL / PPO 训练配置
- `scripts/rsl_rl/train.py`：策略训练
- `scripts/rsl_rl/play.py`：加载策略并进行可视化运行
- `scripts/evaluation/evaluate_policy.py`：独立策略评估
- `results/evaluation/`：定量评估结果
- `results/figures/`：实验结果图
- `results/videos/`：Manipulation 成功案例与 Failure Case 演示

## 项目结果

实验数据、结果图与演示视频统一保存在：

`results/`

其中包括：

- Quantitative Evaluation Results
- Manipulation Success Cases
- Baseline Drop Failure
- 其他 Failure Cases

---

## 当前进度

- [x] Franka Manipulation Baseline
- [x] PPO Parallel Training
- [x] Staged Reward Design
- [x] Reward Hacking 分析与修正
- [x] Multi-Seed Training
- [x] Independent Evaluation
- [x] Experiment 1：Reward Design
- [ ] Experiment 2：Position Generalization
