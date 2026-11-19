# VideoRec · 一次短视频曝光后，用户会点赞吗？

在 KuaiRand-Pure 的 50,000 次真实曝光上，比较逻辑回归、DeepFM 和多任务模型对点赞的预测。每个模型都用过去的数据训练，再预测后续曝光。十种设置各跑三个种子，按验证集点赞 LogLoss 选出小容量 Shared-Bottom。它在测试集上的 AUC 从逻辑回归的 **0.620100 升至 0.664556**，而 AP 从 **0.068544 降至 0.065612**。[验证选择记录](docs/evidence/stage_20261117_resume/selection.json) · [完整指标](docs/evidence/stage_20261117_resume/all_metrics.csv)

## 从数据到模型选择

| 实验要回答的问题 | 做法与观察 | 可核查的记录 |
|---|---|---|
| 打分时能知道哪些信息？ | 仅使用曝光时可获得的字段，假设视频素材时长可取。按时间切成 36,422 / 6,813 / 6,765 行，编码器只用训练期数据拟合。测试期有 124 次点赞。 | [数据处理](rec_lab/data.py) · [切分与正例数](docs/evidence/stage_20261117_resume/data_summary.json) |
| 增加模型复杂度值得吗？ | 从常数和逻辑回归开始。在同一划分比较特征交互、共享层和任务门控，再改变共享层容量、用户 ID 或辅助损失权重。十种设置均保留结果。 | [受控实验配置](scripts/reproduce_stage.py) · [模型实现](rec_lab/models.py) · [30 次运行状态](docs/evidence/stage_20261117_resume/status.json) |
| 测试 AUC 最高的 MMoE 为什么没入选？ | 选择规则在验证集上执行。小容量 Shared-Bottom 的验证 LogLoss 为 0.079908，优于 MMoE 的 0.080526。测试表在选择完成后用于报告。 | [冻结的选择记录](docs/evidence/stage_20261117_resume/selection.json) · [训练与验证](rec_lab/runner.py) |
| 这个优势有多稳？ | 将同一用户的所有测试曝光成组抽取。对 5,284 名测试用户做 1,000 次配对重采样，AUC 差值的 95% 区间为 [0.005575, 0.079841]；LogLoss 和 AP 的区间都跨零。 | [抽样结果](docs/evidence/uncertainty_20261119/summary.json) · [重采样记录](docs/evidence/uncertainty_20261119/run.json) |

这些区间描述固定预测在本次测试窗口的波动。AUC 上升说明这批曝光中的正负样本整体更容易排开。头部正例质量、其他时间窗口的表现、上线后是否增加点赞，各自需要不同的证据。

## 复现第一条 LR 基线

新检出仓库后，在 Windows PowerShell 的仓库根目录运行。下载器从官方 Zenodo 获取数据并校验 MD5，预处理器生成固定日期的 50k 曝光样本。已有环境和处理数据时，从第二个代码块开始。[KuaiRand 数据来源与许可](https://github.com/chongminggao/KuaiRand)

~~~powershell
py -3.12 -m venv .venv_repro
$p = '.venv_repro/Scripts/python.exe'
& $p -m pip install -r requirements-stage-lock.txt
& $p -m pip check
& $p scripts/fetch_kuairand.py
& $p -m scripts.prepare_stage
~~~

~~~powershell
$p = '.venv_repro/Scripts/python.exe'
$id = 'lr_' + (Get-Date -Format yyyyMMdd_HHmmss)
& $p -m scripts.reproduce_stage --mode one --run-id $id
& $p -m scripts.verify_stage --suite "runs/$id"
Get-Content "runs/$id/summary.csv"
~~~

测试集点赞一行应显示 AUC 约 0.620100、LogLoss 约 0.088893。核验程序检查时间边界、源数据行隔离、同一曝光的反馈字段、未知 ID，以及保存后重新加载的模型能否复现验证预测。项目测试可运行 `& $p -m unittest discover -s tests -q`。[核验代码](scripts/verify_stage.py) · [新检出环境验证](docs/evidence/checkout_tests_20261117.txt)

## 实现与复盘

项目代码覆盖时间切分、特征编码、PyTorch 模型、验证选择和评估。逻辑回归求解与张量自动求导使用成熟库，开发和文档整理使用了 AI 辅助。仓库保留源码、配置和聚合实验结果，原始数据按来源条款单独获取。研究范围为离线曝光反馈预测，召回和线上分流属于后续工作。


## 阅读与复现

[40 分钟技术面讲稿 · PDF](docs/VideoRec-Interview-40min.pdf) · [可编辑讲稿](docs/INTERVIEW_40MIN.md) · [固定验收范围](docs/PROGRESS.md)

本公开版本取自开发快照 `197cb49bbbd70838fce1efb070ced924179a9f50`。实验文件记录当时的代码版本、数据指纹与运行环境。公开仓库从本次发布开始记录提交。
