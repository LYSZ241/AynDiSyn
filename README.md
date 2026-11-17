# AynDiSyn

## 项目结构

```text
gcn/                 统一穴位预测入口
R01/                 主穴、配穴双输出入口
common/              数据加载、训练与评价
datasets/numeric/    883条编号记录及五折划分
tests/               软件测试
fixtures/synthetic/  测试样例
tools/               样例生成工具
```

`gcn/` 和 `R01/` 均以 `train_baseline_myfold.py` 为训练入口，参数定义见 `parser.py`，模型定义见 `gcn_model3.py`。

## 安装

使用 Python 3.9 或更新版本，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

## 运行

默认读取 `datasets/numeric/`，`--fold 0` 使用第一折：

```bash
python gcn/train_baseline_myfold.py --fold 0 --epoch 100 --device cpu --seed 2026
python R01/train_baseline_myfold.py --fold 0 --epoch 100 --device cpu --seed 2026
```

使用GPU时将 `--device cpu` 改为 `--device cuda:0`。运行全部五折：

```bash
python R01/train_baseline_myfold.py --all_folds 1 --epoch 100 --device cuda:0 --seed 2026
```

可通过 `--batch_size`、`--lr`、`--embed_size` 和 `--Ks "[5,10,15,20]"` 设置批大小、学习率、嵌入维度及评价截断位置。结果保存在 `outputs/`，包含训练损失和 Recall、Precision、NDCG；R01分别输出总体、主穴和配穴指标。添加 `--save_checkpoint 1` 保存末轮参数。

## 数据格式

`gcn_cases.txt` 为症状集合、穴位集合两列；`r01_cases_5col.txt` 为症状集合、证候集合、病名集合、主穴集合、配穴集合五列。列间使用TAB，集合内使用空格，编号从0开始。`folds.json` 保存训练和测试记录的行索引；读取时保留空字段和原有顺序。

## 检查安装

```bash
python -m unittest discover -s tests -v
```
