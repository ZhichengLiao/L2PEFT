# Profiling: Module-Level Importance Scoring

## 目标

给定 base model 和 RL 训练后的 checkpoint，为模型中**每一个参数张量**打分，衡量它在 RL 训练中的变化程度。打分结果用于下游的 Selective LoRA（只在重要位置插 adapter）和 Selective Freeze（冻结不重要位置）。

## 核心决策

### 不预排除任何参数

所有参数张量（weight、bias、LayerNorm、embedding）都参与打分和排序。不做先验假设——让数据说话。

### 两种独立的打分方法

我们发现"一个 module 变了多少"可以从两个正交维度衡量：

**修改广度（Fraction of Changed Elements）**

```
score = count(|delta_ij| > threshold) / numel
```

逐元素判断是否超过阈值，然后数比例。度量的是"这个参数张量有多大比例的元素被 RL 碰过"。

- 与已有的元素级稀疏性分析框架一致
- 天然是比例，不受矩阵大小影响
- 在 fp32 checkpoint 下，1e-5 阈值能干净地分开"改了"和"没改"

**能量集中度（Relative L2 Change）**

```
score = ||delta||₂ / (||base||₂ + eps)
```

度量的是"变化的总能量占原始权重总能量的比例"。

- 与之前 heatmap (summary_03) 使用的指标一致
- 会被少数大变化元素主导

**两者的区别：** 一个矩阵可能只有 5% 的元素变了（fraction 低），但那 5% 变化剧烈（L2 高）——这是"集中型修改"。反过来，80% 元素都变了但每个只动了一点（fraction 高，L2 低）——这是"广泛型修改"。

两种方法分别生成 mask，比较排序差异本身就是一个研究发现。

### 精度

- 训练：bf16 计算，fp32 master weights
- Checkpoint 存储：fp32
- 加载和打分计算：全部 fp32

Delta 的精度由 fp32 决定，远高于 bf16 量化步长。

### 输出格式

两个脚本输出相同格式的 JSON：

```json
{
  "method": "fraction | l2",
  "top_k_percent": 20,
  "all_scores": {"param_name": score, ...},
  "active_params": ["..."],
  "frozen_params": ["..."],
  "metadata": {
    "base_model": "...",
    "checkpoint": "...",
    "total_params_scored": 196,
    "active_count": 39,
    "score_range": {"max": ..., "min": ..., "active_threshold": ...}
  }
}
```

`all_scores` 包含**全部**参数的分数，不只是 active 的。

## 文件

```
profiling/
├── utils.py                    # 共享：模型加载、delta 计算、mask 输出
├── generate_mask_fraction.py   # 修改广度打分
└── generate_mask_l2.py         # 能量集中度打分
```

## 用法

```bash
# 修改广度
python generate_mask_fraction.py \
    --base_model /path/to/base \
    --checkpoint /path/to/ckpt \
    --threshold 1e-5 \
    --top_k_percent 20 \
    --output mask_fraction.json

# 能量集中度
python generate_mask_l2.py \
    --base_model /path/to/base \
    --checkpoint /path/to/ckpt \
    --top_k_percent 20 \
    --output mask_l2.json
```

## 下游使用

1. **Selective LoRA**：从 mask 的 active_params 中提取 linear module 位置，转为 PEFT `target_modules` 的 regex 列表
2. **Selective Freeze**：遍历 frozen_params，对对应参数设 `requires_grad=False`（在 FSDP wrap 前）
3. **分析**：比较两种方法的排序差异，研究"能量集中"vs"修改广度"的关系
