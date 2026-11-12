# AynDiSyn：完整编号数据与投稿期间的运行示例

仓库提供研究数据的**完整883条编号记录**，没有抽样；不提供编号与临床名称的对照。`gcn/`、`R01/` 保留六文件结构，但核心算法暂以通用基线替代，**不是论文完整实现，不能复现论文指标**。

## 完整编号数据

默认输入目录为 `datasets/numeric/`。两个TXT表示同一批883条记录，行顺序一致：

| 文件 | 内容 |
|---|---|
| `gcn_cases.txt` | 症状编号集合、穴位编号集合，两列以TAB分隔 |
| `r01_cases_5col.txt` | 症状集合、证候集合、病名集合、主穴集合、配穴集合，五列以TAB分隔 |
| `cases.json` | 同样883条记录的纯编号结构，仅含六类编号数组 |
| `folds.json` | 原有五折训练/测试行索引，未重新划分 |
| `train_fold_0.txt` 至 `train_fold_4.txt` | 五折训练数据，两列编号格式 |
| `test_fold_0.txt` 至 `test_fold_4.txt` | 五折测试数据，两列编号格式 |
| `manifest.json` | 数据规模、编号基数与格式信息，不含实体名称 |

编号空间包含2296个症状、303个穴位、188个病名和437个证候。各类型分别从0编号，主穴与配穴共用同一穴位编号空间。症状、穴位和主配标签沿用源数据编号与集合内顺序；病名、证候按源词典中的索引导出。原始病例标识和名称映射不发布。

五折训练/测试数量依次为706/177、706/177、706/177、707/176、707/176；每条记录恰好出现于一个测试折。保留原有2条配穴为空的记录；TXT末列为空时仍有末尾TAB，不要将其当成缺列或删行。数据格式细节见 `datasets/numeric/README.md`。

## 运行

建议Python 3.9或更新版本；先在独立环境安装兼容本机的PyTorch，再安装依赖。默认读取完整编号数据，以下命令只是试跑通用示例：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python gcn/train_baseline_myfold.py --epoch 2 --device cpu
python R01/train_baseline_myfold.py --epoch 2 --device cpu
```

五折运行：

```bash
python gcn/train_baseline_myfold.py --all_folds 1 --epoch 2 --device cpu
python R01/train_baseline_myfold.py --all_folds 1 --epoch 2 --device cpu
```

参数包括 `--data_path`、`--fold`、`--all_folds`、`--epoch`、`--batch_size`、`--lr`、`--embed_size`、`--seed`、`--device`、`--Ks`。结果写入被Git忽略的 `outputs/`；`--save_checkpoint 1` 保存示例的末轮参数。

## 示例模型与评价

GCN目录中的替代模型使用症状编号Embedding、均值池化和线性输出，以普通BCE学习穴位并集。R01目录中的替代模型使用同样的输入表示、两个独立线性头，主穴与配穴BCE等权相加。诊断与证候编号保留在数据中，但通用替代模型不使用它们推理。

评价报告Recall、Precision、NDCG的@5/@10/@15/@20。R01总体分数取两个分支的逐项最大值，主配分别评价；并列分数按编号升序排序。Precision分母固定为K，Recall/NDCG跳过相应标签集为空的记录；该角色全部为空时显示N/A。运行记录明确标注通用示例，不能当作论文实验结果。

两个目录各保留 `train_baseline_myfold.py`、`gcn_model3.py`、`load_data5_myfold.py`、`parser.py`、`dataloader.py` 和 `batch_test.py`。`common/` 为新写的通用加载、训练、评价代码。

## 保留的合成测试样例

`fixtures/synthetic/` 中原有100条是从零生成的**虚构测试样例**，不是883条数据的抽样，也不是默认训练数据。它们只用于软件测试，可以用 `python tools/generate_synthetic.py` 重新生成；此命令不会修改完整编号数据。

## 暂不发布的内容

不发布实体名称词典（包括 `symptom.txt`、`acupoint.txt`、病名/证候词典和 `vocab.json`）、病例原文、源病例标识、真实图与知识矩阵、研究权重和实验日志。多图融合/GAT、层级HGCN、共享角色投影、研究推荐编码器和专用损失也暂不发布。范围见 `RELEASE_SCOPE.md`，后续完整实现由作者另行决定。
