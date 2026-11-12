# AynDiSyn：投稿期间的可运行示例

此仓库当前只发布**接口与训练流程示例**。`gcn/`、`R01/` 保留熟悉的脚本名称，但模型已替换为通用基线，**不是论文完整实现，不能复现论文指标**。

原研究中的图构建与融合、图神经网络、层级超图传播、角色参数共享、推荐编码器、专有损失及知识辅助模块均未公开。真实病例、真实病例的编号版、实体词典、模型权重和研究日志也未公开。后续完整实现与数据的发布由作者另行决定。

## 运行

建议 Python 3.9 或更新版本。先在独立环境安装兼容本机的 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python gcn/train_baseline_myfold.py --epoch 2 --device cpu
python R01/train_baseline_myfold.py --epoch 2 --device cpu
```

五折示例：

```bash
python gcn/train_baseline_myfold.py --all_folds 1 --epoch 2 --device cpu
python R01/train_baseline_myfold.py --all_folds 1 --epoch 2 --device cpu
```

参数包括 `--fold`、`--all_folds`、`--epoch`、`--batch_size`、`--lr`、`--embed_size`、`--seed`、`--device` 和 `--Ks`。默认仅跑 fold0；各折训练80条、测试20条。输出目录默认是 `outputs/`，不加入版本控制；添加 `--save_checkpoint 1` 可保存该示例的末轮参数。

## 合成编号示例

`fixtures/synthetic/` 的100条样本是代码从零生成的演示数据，与真实病例没有对应关系，不保留真实分布或实体含义。48个输入编号、24个候选输出编号、8个诊断编号、12个证候编号都是虚构索引。编号没有公开或隐藏的临床名称映射。

- `gcn_cases.txt`：`症状编号集合\t穴位编号集合`。
- `r01_cases_5col.txt`：`症状编号集合\t证候编号集合\t病名编号集合\t主穴编号集合\t配穴编号集合`。
- 集合内部以空格分隔；索引从0开始。两份文件按行对应，GCN的输出集合等于R01的主配并集。
- `manifest.json` 仅记录合成属性、规模和生成种子；`folds.json` 保存同一套五折行号。

重新生成完全相同的示例：

```bash
python tools/generate_synthetic.py
```

生成规则是人为设置的整数模运算与随机采样，只为演示可学习的输入输出接口，**没有医学解释，不能用于诊疗或报告临床效果**。

## 示例模型和评价

GCN目录中的替代模型将输入编号嵌入取均值，经线性层输出24个分数，使用普通二元交叉熵。R01目录中的替代模型共享同样的均值输入表示，通过两个独立线性层输出主、配分数；两个普通二元交叉熵相加，等权1:1。诊断与证候列只用于展示数据格式，示例模型不使用它们进行推理。

评价按病例计算后取平均，报告 Recall、Precision、NDCG 的 @5/@10/@15/@20。R01总体榜使用两个分支分数逐项取最大值，主配榜独立评价；并列分数按编号升序排序。Precision的分母固定为K，Recall与NDCG在该标签集为空时跳过该例。所有分数都仅为合成示例结果。

## 文件布局

两个目录各保留六个入口文件：

| 文件 | 作用 |
|---|---|
| `train_baseline_myfold.py` | 命令行训练入口 |
| `gcn_model3.py` | 已公开的通用替代模型 |
| `load_data5_myfold.py` | 编号样本与五折加载 |
| `parser.py` | 示例参数 |
| `dataloader.py` | 通用批处理接口 |
| `batch_test.py` | 通用评价接口 |

`common/` 仅包含从零编写的通用加载、训练和指标代码，没有隐藏的研究核心实现。发布范围见 `RELEASE_SCOPE.md`。
