# Covariate-dependent Graphical Model (AGENTS.md)

本文档遵循 [AGENTS.md](https://agents.md) 标准，旨在为 AI 代理和人类开发者提供关于本项目的权威指南。本项目实现了论文 **"Covariate-dependent Graphical Model Estimation via Neural Networks with Statistical Guarantees"** 中的核心算法。

## 理论基础 (Theoretical Foundation)

本项目旨在解决协变量依赖的图模型估计问题。我们的目标是根据协变量 $Z \in \mathbb{R}^q$ 估计节点变量 $X \in \mathbb{R}^p$ 的条件精度矩阵 $\Omega(Z) = \Sigma(Z)^{-1}$。

### 核心数学模型：节点级回归

该方法通过节点级回归来恢复图结构。对于图中的每个节点 $i \in \{1, \dots, p\}$，我们将其作为响应变量，其余节点 $X_{-i}$ 作为预测变量。模型假设如下线性关系（以 $Z$ 为条件）：

$$
X_i = \sum_{j \neq i} \beta_{ij}(Z) X_j + \epsilon_i
$$

其中：
*   $X_i$ 是第 $i$ 个节点的观测值。
*   $\beta_{ij}(Z)$ 是依赖于协变量 $Z$ 的回归系数，量化了节点 $j$ 对节点 $i$ 的条件依赖强度。
*   $\epsilon_i$ 是均值为零的噪声项。

### 神经网络参数化

为了捕捉 $\beta_{ij}(Z)$ 的非线性依赖关系，我们使用神经网络对其进行建模：

$$
\hat{\beta}_{ij}(Z) = \text{NN}_{\theta}(Z)_{ij}
$$

在代码中，这通过一个多层感知机 (MLP) 或残差网络 (ResNet) 实现，输入为 $Z$，输出为所有成对系数的向量。

### 优化目标

模型通过最小化均方误差 (MSE) 进行训练，并结合正则化项（如 $\ell_1$ 或 $\ell_2$ 范数）以鼓励稀疏性或平滑性：

$$
\min_{\theta} \sum_{k=1}^{n} \sum_{i=1}^{p} \left( X_i^{(k)} - \sum_{j \neq i} \beta_{ij}(Z^{(k)}; \theta) X_j^{(k)} \right)^2 + \lambda \mathcal{R}(\theta)
$$

## 代码与理论映射 (Codebase Mapping)

本节提供了代码变量与论文数学符号的严格对应关系，以及关键逻辑的文件位置。

### 术语对齐 (Terminology Alignment)

| 数学符号 (Math Symbol) | 代码变量 (Code Variable) | 对应文件 (File Context) | 描述 (Description) |
| :--- | :--- | :--- | :--- |
| $X$ (Nodes) | `x` / `data` | `models/networks.py` | 节点变量矩阵，形状为 `[batch_size, num_nodes]` |
| $Z$ (Covariates) | `z` | `models/networks.py` | 协变量矩阵，形状为 `[batch_size, num_covariates]` |
| $\beta_{ij}(Z)$ | `beta_network` 的输出 | `models/networks.py` (`dnnCGM` 类) | 神经网络根据 $Z$ 预测的回归系数 |
| $p$ (Number of nodes) | `num_nodes` | `configs/*.yaml` | 图中节点的数量 |
| $q$ (Dim of covariates) | `num_covariates` / `dim_z` | `configs/*.yaml` | 协变量的维度 |
| $\mathcal{L}$ (Loss Function) | `criterion = nn.MSELoss()` | `utils/utils_train.py` | 均方误差损失函数 |
| $\Omega(Z)$ (Precision Matrix) | `graph` / `symSkel` | `run_sim.py` (输出结果) | 最终估计的图结构或精度矩阵 |

### 文件结构上下文 (File Structure Context)

*   **模型定义**: `models/networks.py`
    *   类 `dnnCGM`: 实现了核心的节点级回归逻辑。
    *   函数 `forward(self, x, z)`: 计算 $\hat{X} = \beta(Z) \cdot X$，对应公式 $X_i = \sum \beta_{ij}(Z) X_j$。
*   **优化过程**: `utils/utils_train.py`
    *   函数 `train_epoch`: 执行单次训练迭代，计算 MSE Loss 并更新梯度。
    *   函数 `run_model_pipeline`: 管理整个训练流程，包括优化器初始化。
*   **数据生成**: `generator/simulator/sim_GGM.py`
    *   类 `GGM_Simulator`: 根据预设的 $\Omega(Z)$ 生成合成数据，用于验证模型性能。

## 开发环境提示 (Dev environment tips)

### 安装与依赖 (Prerequisites & Dependencies)

本项目依赖以下核心数学和深度学习库。根据 `setup.py`，建议使用以下版本：

*   **Python**: >= 3.10
*   **PyTorch** (`torch`): `==2.2.2` (用于构建神经网络 $\beta(Z)$ 和自动微分)
*   **NumPy** (`numpy`): `==1.26.4` (用于矩阵运算)
*   **SciPy** (`scipy`): `==1.14.1` (用于科学计算)
*   **Scikit-learn** (`scikit_learn`): `==1.5.2` (用于数据预处理和评估)
*   **Matplotlib** (`matplotlib`): `==3.9.2` (用于绘图)
*   **Seaborn** (`seaborn`): `==0.12.2` (用于高级统计绘图)
*   **NetworkX** (`networkx`): `==3.1` (用于图论计算和可视化)
*   **PyYAML** (`PyYAML`): `==6.0.2` (用于配置文件解析)
*   **TorchMetrics** (`torchmetrics`): `==1.4.1` (用于模型评估指标)
*   **TensorBoard** (`tensorboard`): `==2.18.0` (用于训练过程可视化)
*   **JupyterLab** (`jupyterlab`): `==4.4.0` (用于运行 Notebook 示例)
安装命令：
```bash
pip install -e .
```


### 数据生成

如果需要重新生成合成数据：

有效的 `DATA_STR`: `GGM1`, `GGM2`, `NPN1`, `NPN2`, `DAG1`, `DAG2`。

```bash
# 生成高斯数据 (Gaussian data)
python -u generator/generate_GGM.py --data_str=$DATA_STR --data_seed=$DATA_SEED

# 生成基于 DAG 的数据 (DAG-based DGP)
python -u generator/generate_DAG.py --data_str=$DATA_STR --data_seed=$DATA_SEED

# 生成非正态数据 (Non-paranormal based DGP)
python -u generator/generate_NPN.py --data_str=$DATA_STR --data_seed=$DATA_SEED
```
参见 `configs/_synthetic_.yaml`。

要一次性生成多个数据副本：
```bash
cd bin
bash generate.sh $DATA_STR
```

### 运行实验

#### 1. 单次实验 (Python 脚本)
使用 `run_sim.py` 脚本运行完整的训练和评估流程。

```bash
# VERSION 为空（默认，神经网络方法）或 `reggmm`（线性方法）
python -u run_sim.py --data_str=$DATA_STR --train_size=$TRAIN_SIZE --gpu=$GPU_ID --data_seed=$DATA_SEED --version=$VERSION
```

#### 2. 批量实验 (Shell 脚本)
项目在 `bin/` 目录下提供了自动化脚本，可用于复现论文中的大规模实验：

*   `bin/run.sh`: 运行单个设置的完整流程。
*   `bin/run-all.sh`: 运行所有数据集配置的实验。
*   `bin/run-r.sh`: 运行使用 R 包（glasso 或 neighborhood selection）的竞争模型。

```bash
# 运行多个合成数据副本
cd bin
bash run.sh $DATA_STR $GPU_ID $VERSION
```

最后，要运行任何特定数据设置（通过 `$DATA_STR` 指定）的整个流程，包括生成数据、运行提出的 DNN 模型和竞争者，请参考 `bin/run-all.sh`。

## 真实数据实验 (Real Data Experiments)

除了合成数据，本项目还提供了处理真实数据的示例和脚本。

### 1. 静息态 fMRI 数据 (Resting-state fMRI Dataset)
*   **数据转换**: 原始数据通常为 `.RData` 格式。本项目提供了 R 脚本 `data/convert-fmri-data.R`，利用 `reticulate` 包将其转换为 Python 可读的 `.npz` 格式。
*   **演示 Notebook**: 参见 `notebooks/demo-run-fmri-data.ipynb`，包含端到端模型运行和结果可视化。

### 2. S&P 100 成分股数据 (S&P 100 Constituents Dataset)
*   **数据位置**: 处理后的数据位于 `data/stock/` 目录下。
    *   $X$ (节点值): 成分股的 $\beta$ 调整后残差收益率（相对于 SPX）。
    *   $Z$ (协变量): SPX, Nasdaq 收益率及 VIX 水平。
*   **演示 Notebook**: 参见 `notebooks/demo-run-stock-data.ipynb`。

## 测试指南 (Testing instructions)

*   **运行测试**: 目前主要通过运行 `run_sim.py` 并检查 `output_sim/` 目录下的结果来进行验证。
*   **配置验证**: 所有实验配置均位于 `configs/` 目录下。修改 YAML 文件中的 `n_nodes` 或 `sparsity` 可改变实验的数学设定，并观察结果变化。
*   **调试模式**: 使用 `--debug` 标志运行脚本以启用详细日志记录和中间输出检查。

## 贡献指南 (PR instructions)

*   **代码风格**: 请遵循 PEP 8 编码规范。建议在提交前使用 linter（如 `flake8`）检查代码。
*   **提交前验证**: 在提交 Pull Request 之前，请务必运行一个小规模的实验（例如 `GGM1` 设置），确保核心训练流程（`run_sim.py`）能够正常完成且无报错。
*   **新功能**: 如果添加了新的模拟器或模型架构，请在 `configs/` 中添加相应的配置文件，并在 `AGENTS.md` 的“代码与理论映射”部分更新相关说明。

## 教学示例 (Educational Examples)

以下代码片段展示了如何直接操作模型的核心数学参数，观察输入 $X, Z$ 如何转化为预测值 $\hat{X}$。这有助于理解 $\beta_{ij}(Z)$ 的作用机制。

```python
import torch
from models.networks import dnnCGM

# 1. 定义超参数 (Hyperparameters)
# 对应论文中的 p (节点数) 和 q (协变量维度)
configs = {
    'num_nodes': 5,          # p
    'num_covariates': 3,     # q
    'beta_module_name': 'MLP',
    'beta_hidden_dims': [10, 10], # 神经网络架构参数
    'beta_dropout': 0.1,
    'beta_batch_norm': True,
    'beta_bias': True
}

# 2. 初始化模型
# 这将创建神经网络 \beta(Z)
model = dnnCGM(configs)
print(f"模型已初始化，参数量: {model.count_num_params()}")

# 3. 创建模拟数据
batch_size = 4
# X: [Batch, p] - 节点观测值
x_dummy = torch.randn(batch_size, configs['num_nodes']) 
# Z: [Batch, q] - 协变量
z_dummy = torch.randn(batch_size, configs['num_covariates'])

# 4. 前向传播 (Forward Pass)
# 模型内部执行步骤:
# a. 计算系数: beta = beta_network(z)
# b. 重构 X: x_hat = beta * x
x_pred = model(x_dummy, z_dummy)

print(f"输入 X 形状: {x_dummy.shape}") # [4, 5]
print(f"输入 Z 形状: {z_dummy.shape}") # [4, 3]
print(f"预测 X 形状: {x_pred.shape}")   # [4, 5]

# 5. 计算损失 (Loss Calculation)
# 对应公式: || X - X_hat ||^2
loss = torch.nn.MSELoss()(x_pred, x_dummy)
print(f"当前均方误差 (MSE): {loss.item()}")
```
