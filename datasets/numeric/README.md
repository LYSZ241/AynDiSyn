# 完整883条编号数据

每行对应一条原研究记录；行序、已有症状/穴位编号、处方标签和五折划分均保留。这里没有抽样、补充或清洗操作。

## TXT

`gcn_cases.txt` 与各 `train_fold_N.txt`、`test_fold_N.txt` 的两列为：

```text
symptom_ids<TAB>acupoint_ids
```

`r01_cases_5col.txt` 的五列为：

```text
symptom_ids<TAB>syndrome_ids<TAB>disease_ids<TAB>main_ids<TAB>adjunct_ids
```

`<TAB>`在文件中是实际制表符；集合内以空格分隔，无表头。主穴与配穴的集合并集等于穴位集合，但不得据此重新排序源穴位列。2条配穴为空的记录保留空末列；应先去掉换行，再按TAB拆分，不能对整行调用 `strip()`。

## 编号与JSON

所有编号从0开始：症状0–2295，穴位0–302，病名0–187，证候0–436。各类型编号空间独立，不能把相同整数当成相同实体。主穴和配穴共用穴位编号空间，角色由每条记录中的字段表示。

`cases.json`仅包含 `symptom_ids`、`acupoint_ids`、`main_ids`、`adjunct_ids`、`disease_ids`、`syndrome_ids` 六个整型数组字段，未附加姓名、临床原文或源病例标识。病名与证候按源词典位置转为编号，不提供词典或反向映射。

`folds.json`为五个对象组成的列表；每个对象的 `train` 和 `test` 是上述883条记录的0基行索引。训练/测试依次为706/177、706/177、706/177、707/176、707/176，测试集覆盖全部883条且不重叠。各折TXT顺序与索引数组一致；主配评估标签通过同一行索引从五列TXT或 `cases.json` 取得。

`manifest.json`仅记录规模、格式和分折信息，不包含实体名称。为保留原编号空间，规模取源词典大小，不重新压缩或编号。
