"""Build the factual handoff from an existing completed experiment, not invented results."""
import csv
import json
import shutil
from pathlib import Path
import pandas as pd
from rec_lab.data import sha256, write_json
from rec_lab.runner import read_json

ROOT=Path(__file__).resolve().parents[1]
RUN='stage_20261107_full'
DATA='data/processed/real_stage_20261107'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])

def write(name,text):
    (ROOT/'docs'/name).write_text(text.strip()+'\n',encoding='utf-8')

def main():
    out=ROOT/'runs'/RUN; m=read_json(ROOT/DATA/'manifest.json'); env=read_json(out/'environment.json')
    status=read_json(out/'status.json'); selection=read_json(out/'selection.json')
    assert status['state']=='completed'
    summary=pd.read_csv(out/'summary.csv'); comp=pd.read_csv(out/'comparison.csv')
    run_dirs=[out/name for name in status['runs']]
    rss=max(read_json(p/'resources.json')['sampled_peak_process_rss_bytes'] for p in run_dirs)
    smoke=read_json(ROOT/'runs/stage_20261107_smoke/single_s42/config.json')
    primary=summary.query("split=='test' and target=='is_like'")
    test_table=table(['模型','测试 AUC 均值 ± SD','测试 LogLoss 均值 ± SD','参数量'],[
        [r.experiment,f'{r.auc_mean:.6f} ± {r.auc_std:.6f}',f'{r.logloss_mean:.6f} ± {r.logloss_std:.6f}',r.parameters]
        for r in primary.itertuples()])
    split_table=table(['集合','行数','点赞数 / 正例率','用户 / 视频','未见用户 / 视频曝光比例','时间范围（上海）'],[
        [k,v['rows'],f"{v['is_like_positive_count']} / {v['like_rate']:.4%}",f"{v['users']} / {v['videos']}",
         f"{v['unseen_user_row_rate']:.2%} / {v['unseen_video_row_rate']:.2%}",f"{v['time_min_shanghai']} — {v['time_max_shanghai']}"] for k,v in m['split_audit'].items()])
    groups=[]
    for p in run_dirs:
        g=read_json(p/'test_groups.json')['is_like']
        for name in ['seen_video','unseen_video','seen_user','unseen_user']:
            if name in g:groups.append(dict(run=p.name,experiment=p.name.rsplit('_s',1)[0],group=name,**{k:v for k,v in g[name].items() if k!='calibration'}))
    gf=pd.DataFrame(groups);gf.to_csv(out/'group_comparison.csv',index=False)
    gs=gf.groupby(['experiment','group']).agg(rows=('rows','first'),positive_count=('positive_count','first'),auc=('auc','mean'),logloss=('logloss','mean')).reset_index()
    gt=table(['模型','分组','样本数','点赞数','三种子平均 AUC','平均 LogLoss'],[
        [r.experiment,r.group,r.rows,r.positive_count,'不可计算' if pd.isna(r.auc) else f'{r.auc:.6f}',f'{r.logloss:.6f}']
        for r in gs[gs.experiment.isin(['lr','single','shared_aux0p0','shared_aux0p2','shared_aux1p0']) & gs.group.isin(['seen_video','unseen_video'])].itertuples()])
    raw=ROOT/'data/raw';archive=raw/'KuaiRand-Pure.tar.gz'
    rawfiles=[dict(relative=str(p.relative_to(ROOT)),absolute=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in (raw/'kuairand_extracted').rglob('*') if p.is_file()]
    provenance=dict(verified_date='2026-11-07 Asia/Shanghai',official_page='https://kuairand.com/',
        record='https://zenodo.org/records/10439422',version='Zenodo v1, published 2024-01-01',
        archive=dict(name=archive.name,bytes=archive.stat().st_size,md5='0820331067a3784d9691136f772b35a7',sha256=sha256(archive)),files=rawfiles,
        use='Local learning and offline experiments; cite Gao et al., CIKM 2022. Repository LICENSE is CC-BY-SA-4.0; Zenodo parsed license was blank, do not infer unrestricted commercial redistribution rights.')
    write_json(raw/'provenance_stage.json',provenance)
    rawtable=table(['原始文件（解压 data/）','字节数','用途'],[[Path(x['relative']).name,x['bytes'],'训练来源' if 'log_standard' in x['relative'] else '保留但不进入本轮模型'] for x in rawfiles])
    write('EXPERIMENT_REPORT.md',f'''
# 真实数据受控实验报告（2026-11-07运行，11-08交接）

完成目录：`{out}`。本轮是本机新运行，旧 `examples/real_runs` 仍保留为历史示例。

## 研究问题与预先设定

曝光前预测点赞：增加同一曝光的辅助反馈训练任务，能否改善后续时段的点赞预测？仓库已有静态特征多任务实现，故延续 Shared-Bottom（共享底层、两个任务输出头），不添加历史模型或 MMoE。

假设：辅助任务有助于共享表示。唯一主要改动：相同 Shared-Bottom 结构的辅助权重 α=0/0.2/1。损失 `BCE_like + α × BCE_click`；BCE 是二分类交叉熵。主任务系数保持1。所有神经网络统一按验证集点赞 LogLoss 选择最佳轮次，最多12轮、patience=3，batch=512，Adam学习率0.001、weight_decay=1e-5、梯度裁剪5、Embedding8维、隐藏层32/16。种子42/43/44，数据、batch次序规则、初始化规则和优化器一致。

容量对照：single 有154145参数；shared有154162参数，多17个辅助头参数。single和shared的点赞头初始化也不同，因此主要机制判断必须看相同shared结构的α对照。α=0仍保留辅助头，其辅助预测没有实际学习价值。

判断依据：三种子平均验证点赞LogLoss，AUC次要。所有设置在测试前写入 [protocol.json](../runs/{RUN}/protocol.json)；[selection.json](../runs/{RUN}/selection.json) 在最终测试前冻结，选中 `{selection['selected_model']}`，shared内选中 `{selection['selected_shared']}`。文件中 `test_not_evaluated_yet=true` 记录的是冻结时状态，最终完成状态见status.json。本轮随后一次性评估所有预先声明对照，没有按测试结果重新调整模型。

## 数据来源、范围与清洗

已核实[官方KuaiRand页面](https://kuairand.com/)和[Zenodo v1](https://zenodo.org/records/10439422)。下载文件 `{archive.name}`，实际 {archive.stat().st_size} 字节，公开MD5 `0820331067a3784d9691136f772b35a7` 校验通过，SHA256 `{sha256(archive)}`。所有原文件SHA256及大小见 `data/raw/provenance_stage.json`，处理输入SHA见manifest。

{rawtable}

官方 is_like 为点赞反馈；is_click 在双列UI表示点击、单列UI表示有效播放，因此这里只称原始反馈辅助任务，不宣称统一CTR。tab是场景编号，本轮按去重后标准曝光记录量最大选tab=1，不按标签挑选，也不猜测该编号具体UI。标准与随机曝光不混合。Pure只保留候选池视频，序列不完整，本轮不声称用户完整历史建模。

原标准日志 {m['source_rows']} 行，跨所有tab全字段精确重复删除 {m['exact_duplicate_rows_removed']} 行；所选tab去重后 {m['cohort_rows_before_cleaning']} 行，因视频总时长duration_ms=0排除 {m['dropped_invalid_rows']} 行，其余所需字段缺失、无效二元标签、非法ID、越界时间等计数见manifest，本次所选场景这些项为0。原始数据不改写。**不按is_click=0或play_time_ms=0删除正常负例**，实际保留数量见 `preserved_negatives`。重复按全部原始字段比较，不按用户视频组合去重；重复与剔除的原始行ID分别保留。

清洗后在完整时间跨度按行位置无放回随机抽50000条，种子20261107，随后按时间排序。样本共 {m['selected_audit']['users']} 用户、{m['selected_audit']['videos']} 视频、{m['selected_audit']['is_like_positive_count']} 点赞。总体/样本按日分布存于cohort_by_day.csv与sample_by_day.csv；均匀随机抽样也不保证覆盖全平台或每个用户。样本点赞率 {m['selected_audit']['like_rate']:.4%}，清洗总体 {m['clean_cohort_audit']['like_rate']:.4%}。不把连续前50000行称作代表性样本。

固定边界：上海时间2022-04-18 00:00和2022-04-28 00:00；精确毫秒值为 {m['protocol']['train_before_ms']}、{m['protocol']['valid_before_ms']}，同一时间戳绝不跨边界。source_row_id是“原文件名:不含表头的一起始数据行号”。逐行切分存于split_membership.csv，三个CSV与SHA256构成数据身份 `{m['data_id']}`。

{split_table}

各集合均有正负类，可计算指标；验证116、测试124个点赞仍然较少，足够小规模链路验证，不足以断言小幅差异稳定。

## 特征和基线公平性

输入只有user_id/video_id/tab、视频总时长log1p、曝光时钟的小时正余弦。duration_ms是视频固有长度，这里假设日志记录值与曝光前元数据一致；未做版本快照证明，是明确的可用性假设。排除本行点赞/点击/观看时长及其派生值，排除用户表与视频月度汇总表，也不输入随机曝光标记或标签历史。没有标签历史更新，验证/测试反馈不会进入输入。

词表和均值/标准差仅训练集拟合。未知ID：MLP用保留的0索引零向量，LR的未知类别one-hot块全0。LR用稀疏独热编码（每个类别一列）加数值标准化，MLP用可学习Embedding后接非线性层；原始信息相同，但表达能力和参数量不同。常数概率固定为训练点赞率 {m['split_audit']['train']['like_rate']:.10f}。

## 真实结果

下表来自 [summary.csv](../runs/{RUN}/summary.csv)，每一项的3次原值在 [comparison.csv](../runs/{RUN}/comparison.csv)，逐run权威文件为 `metrics.json`。SD为三个种子的样本标准差（ddof=1），不是置信区间。常数与确定性的LR三个种子结果相同，其SD=0不代表泛化没有不确定性。

{test_table}

完整逐次记录（含辅助任务AUC、LogLoss）如下；α=0辅助头指标仅用于完整留痕，不能解释为有效辅助模型：

{table(['run','split','task','AUC','LogLoss'],[[r.run,r.split,r.target,f'{r.auc:.6f}',f'{r.logloss:.6f}'] for r in comp.itertuples()])}

![测试结果图](../runs/{RUN}/test_metrics.png)

事实：验证集single平均LogLoss最低；shared内α=0最低。测试中single AUC高于LR，但LR LogLoss更低；加辅助任务未在本协议下带来主任务一致收益。不能用测试更低的LR LogLoss倒推修改已经冻结的验证选择。

可能原因（推测）：点赞稀疏、反馈任务目标不完全一致、时间分布变化，或共享梯度竞争。权重1提升辅助任务排序但可能损害点赞表示。当前结果不能区分这些原因。下一步应先扩大训练样本或分析训练/验证曲线，并预先登记后续假设；因为本测试已公开，后续不能把同测试集继续调参后的改进当成独立确认。

## 分组与限制

已见视频定义为在训练集出现至少一次；未知定义为训练计数0，完全由训练数据确定。主要分组如下，完整逐种子记录在 `group_comparison.csv` 与各run的test_groups.json。单类别AUC为JSON null/表中不可计算，不填0。

{gt}

测试未知视频只有324条曝光、7个点赞，已见视频6441条、117个点赞。single的未知视频平均AUC约0.7084，高于已见约0.6596，却不能据此说未知视频更容易：未知组只用零向量加其他特征，而且正例非常少，抽样波动很大。对应LogLoss约0.10423，高于已见约0.08847，组间正例率也不同。用户未知曝光占比从验证到测试上升，也提示时间迁移；其原因不能仅靠当前结果确定。现有训练活跃度固定桶是补充输出，不作为本轮主要分组结论。

旧示例已使用同一来源时期。本轮换了抽样和明确边界，仍有数据交叠，所以是工程复现及受控探索，不是全新独立验证。数据仅来自Pure标准曝光的一个场景；排除无效固有时长可能引入偏差；没有线上试验、显著性检验或收益承诺。

数据使用：引用Gao等人KuaiRand（CIKM 2022）。[官方仓库LICENSE](https://github.com/chongminggao/KuaiRand/blob/main/LICENSE)标示CC-BY-SA-4.0；本次Zenodo页面的解析结果未显示明确许可文本，不能自行推断无限制商业再分发授权。此交付保留本机研究数据，迁移代码包不包含原始曝光。

## 验证、资源和复现

既有加新增关键检查28项通过；随后新增“辅助权重0时打乱辅助标签不改变主任务预测”行为检查，6项stage检查通过。日志见verification/tests_stage_utf8_20261107.log、ablation_checks_20261107.log。正式18个run全部在无标签输入上重载预测，与保存的valid预测逐项一致（1e-6相对、1e-7绝对容差），未知ID输出有限，详情verification.json。另外已实际执行单个LR复现命令，目录stage_20261107_reproduce_lr。

本机 {env['cpu'].strip()}，{env['physical_cores']}核/{env['logical_cores']}线程，RAM {env['ram_bytes']/2**30:.2f} GiB，启动可用磁盘 {env['disk_free_bytes']/2**30:.2f} GiB；GPU {env['gpu_inventory']}。环境Python3.12，torch {env['torch']} CPU构建，未用GPU。两轮MLP训练 {smoke['training_seconds']:.3f} 秒；18个正式run加重载/最终评价/汇总用时 {status['wall_seconds']:.3f} 秒，单run训练进程采样最大RSS {rss/2**20:.2f} MiB。采样间隔50ms，可能漏短峰值，也不等于全机占用；数据处理阶段未监测峰值，GPU峰值未测。CPU神经训练设2个torch线程。无需AutoDL；迁移准备见AUTODL_GUIDE.md。

依赖完整锁定requirements-stage-lock.txt；每run有config/manifest/encoder/weights/predictions/metrics/history/resources/status/log，套件还有源码快照及哈希、环境和全部依赖。项目本身无Git，不虚构提交号。复现命令见START_HERE.md。
''')
    write('START_HERE.md',f'''
# 从这里接手

实际项目：`{ROOT}`。Windows本机；外层工程目录仅包含旧环境，聊天初始“项目”目录不是项目根。没有适用AGENTS.md，实际项目无Git。旧examples、原数据和student_work/exercises练习保持原样，数学答案由你维护。

**已完成**：官方真实数据校验 → 50000行 → 全局固定时间切分 → 常数/LR/MLP → Shared-Bottom权重0/0.2/1 → 三种子18次实验 → 冻结后测试、分组和模型重载验证。结论：辅助任务没有稳定改善本轮点赞指标。完整结果在 `runs/{RUN}/`。现在不需要租AutoDL。

建议依次读：本页 → LEARNING_GUIDE.md → EXPERIMENT_REPORT.md → FILE_MAP.md → HANDOFF.md。阶段命令与失败记录在STEP_LOG.md。

## 从VS Code复现一个结果

1. Windows本机打开VS Code，菜单“文件 → 打开文件夹”，选择上面的**内层**video_rec_lab_v2。
2. Python扩展“选择解释器 → 输入解释器路径”，选择 `{ROOT / '.venv_repro/Scripts/python.exe'}`。旧外层.venv312保留但不要用于本轮。
3. 新建PowerShell终端，粘贴下方命令。直接调用解释器，不需要激活脚本。读取已存在真实切分，生成新的LR权重、预测和指标；每次run_id自动带时间，保护旧结果。

```powershell
Set-Location -LiteralPath '{ROOT}'
$env:PYTHONUTF8 = '1'
$recPython = Join-Path (Get-Location) '.venv_repro/Scripts/python.exe'
$runId = 'my_lr_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $recPython -m scripts.reproduce_stage --mode one --run-id $runId
& $recPython -m scripts.verify_stage --suite "runs/$runId"
Get-Content "runs/$runId/lr_s42/metrics.json"
```

预期测试is_like AUC≈0.620100、LogLoss≈0.088893（同依赖CPU）。命令已用 `stage_20261107_reproduce_lr` 实跑；查看 `runs/$runId/summary.csv`。若终端返回非零先看日志，不把未成功执行当结果。

同一台机器/同目录/PowerShell/同解释器重跑全套：`& $recPython -m scripts.reproduce_stage --run-id ('my_full_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))`；读取同一数据，生成18个新run。短跑增加 `--mode smoke`，只训练2轮，不评测试。验证：`& $recPython -m unittest discover -s tests -v`。

需要从原始数据重建时，同一终端运行 `& $recPython -m scripts.prepare_stage --out data/processed/my_rebuilt_001`，读取已校验的解压数据并新建处理版本；然后训练加 `--data data/processed/my_rebuilt_001`。现有默认版本不可覆盖。首次换机器的数据/环境步骤与Linux命令分开写在AUTODL_GUIDE.md。

下一步先完成学习指南第1—3节，在 `learning/math/` 写自己的练习；然后对照报告解释“AUC高但LogLoss未必低”。后续扩样先预登记方案，不继续用本测试调参。
''')
    write('STEP_LOG.md',f'''
# 阶段记录

日期以Asia/Shanghai记录：2026-11-07晚开始，11-08凌晨完成文档。以下实际执行均在Windows本机、项目根 `{ROOT}`、PowerShell；训练解释器 `.venv_repro/Scripts/python.exe`，UTF-8模式 `$env:PYTHONUTF8='1'`。命令中的 `python` 下文均为这个解释器的简写，不能调用系统3.14代替。

## 01 定位与保留

目的：识别项目与历史证据。`rg --files` 找到桌面内层项目；读取README、依赖、rec_lab、scripts、tests、练习与examples结果；检查父级和项目AGENTS，未找到适用文件。实际根无.git；聊天工作区的空master仓库不是该项目。旧示例确有真实指标摘要，但没有原始数据/权重/预测，不能当本机已复现实验。旧.venv312无torch。

修改：新增docs/START_HERE.md和空learning/math目录；代码修改前把runner.py、experiments.py、__main__.py保存到verification/before_stage_20261107/。未写练习答案，未搬迁或复制整套项目。

环境识别：CIM查询被沙箱拒绝，改用Python/psutil、winreg与nvidia-smi读取；见runs/{RUN}/environment.json。桌面项目超出聊天默认可写范围，项目写入使用授权执行。曾遇沙箱读取临时目录权限，授权上下文正常。

## 02 独立环境与官方数据

实际命令：系统Python3.12 `-m venv .venv_repro`；`python -m pip install -r requirements.txt psutil torch --index-url https://pypi.org/simple`。仅新增项目环境，不升级旧环境。`python scripts/fetch_kuairand.py` 从已核实Zenodo入口下载47,432,272字节并验证公开MD5，解压到data/raw/kuairand_extracted。下载源与SHA256见data/raw/kuairand_source.json及provenance_stage.json。

## 03 数据协议与代码修复

新增scripts/prepare_stage.py、tests/test_stage.py；runner.py新增aux_weight/selection_target、分任务验证损失、资源采样、预测source_row_id；experiments.py将新权重加入汇总键；__main__.py增加CLI参数，旧接口默认兼容。

实际命令：`python -m scripts.prepare_stage`。读取两份standard原文件，生成 `{DATA}`，50000行，train/valid/test=36422/6813/6765；点赞=753/116/124。输出和处理规则详见manifest，原始行标识与哈希可追踪。日志verification/prepare_stage_20261107.log。

## 04 风险验证与短跑

第一次 `python -m unittest discover -s tests -v`：28项中1项错误，Windows默认GBK读取UTF-8历史报告失败；保留verification/tests_stage_20261107.log。设置PYTHONUTF8=1重跑，28项通过，5.111秒，日志tests_stage_utf8_20261107.log。

新增scripts/reproduce_stage.py、verify_stage.py、package_stage.py。实际命令：`python -m scripts.reproduce_stage --run-id stage_20261107_smoke --mode smoke`。真实数据2轮MLP训练{smoke['training_seconds']:.3f}秒；保存重载验证通过；没有测试评价。依据短跑决定CPU足够。

## 05 固定方案、正式实验与最终评价

实际命令：`python -m scripts.reproduce_stage --run-id {RUN}`。读取同一真实切分，6配置×3种子，18次全部成功，累计流程{status['wall_seconds']:.3f}秒。外层日志verification/full_stage_20261107.log，每run内train.log。protocol.json先于训练写入，selection.json在最终测试前冻结；统一is_like验证LogLoss选checkpoint。全部18个run重载验证成功，随后一次性测试。结果summary.csv/comparison.csv/metrics.json，不隐藏退步。

## 06 补充消融行为验证与用户复现

新增test_zero_auxiliary_weight_ignores_auxiliary_labels：固定主任务，翻转辅助标签，α=0两次训练主预测完全相同，且best_epoch确实对应主任务验证最小值。实际 `python -m unittest discover -s tests -p test_stage.py -v`：6项通过，日志verification/ablation_checks_20261107.log。只补验证，没有据测试结果修改训练方案。

实际 `python -m pip freeze` 保存requirements-stage-lock.txt；`python -m scripts.reproduce_stage --run-id stage_20261107_reproduce_lr --mode one` 已成功，用于验证START_HERE复现入口。该LR复现不是新调参，不加入三种子均值。

## 07 学习与迁移交付

scripts/build_stage_docs.py从已有metrics生成报告、路径表、来源记录和交接；LEARNING_GUIDE.md提供7步练习；AUTODL_GUIDE.md核查官方说明，云端步骤均标为未运行。`python -m scripts.package_stage` 仅打包代码/文档/锁定依赖到migration/video_rec_stage.zip。没有租用实例或收费操作。

成功run：stage_20261107_smoke、{RUN}、stage_20261107_reproduce_lr。模型训练无失败或中止；工程测试一次编码失败已保留并修复。中途协作代理因会话额度中断，已有暂存文件保留，主流程继续完成；这不属于已完成实验。
''')
    write('HANDOFF.md',f'''
# 下一会话交接（2026-11-08）

实际项目根 `{ROOT}`，Windows11本机，独立解释器 `{ROOT/'.venv_repro/Scripts/python.exe'}`。不是WSL/云端，没有本机外的实验文件需同步。外层.venv312、旧examples和你的练习完整保留。无Git提交，修改备份在verification/before_stage_20261107，运行源码快照在runs/{RUN}/source_snapshot。

已完成：官方Pure下载校验、真实50k处理和固定边界、常数/LR/MLP、同结构shared辅助权重消融、42/43/44三种子、最终测试、seen/unseen分组、关键测试、模型重载预测、独立LR复现实跑、学习/日志/路径/资源/迁移说明。原始data/raw；固定data/processed/real_stage_20261107；正式runs/{RUN}；正式结果summary.csv与各run/metrics.json。

未执行：AutoDL实例/付费/GPU/Linux安装，任何云端性能承诺，完整Pure训练、历史模型、MMoE新增实验、线上效果或统计显著性检验。GPU为RTX3060 Laptop6GB，当前torch CPU构建，所以cuda_available=false正常。数据准备峰值未监测；训练内存采样口径见报告。当前没有阻塞本轮复现的缺失依赖。

下一条命令（Windows本机、PowerShell、上述项目根、项目解释器）读取已有数据，另建结果，不覆盖旧结果：

```powershell
Set-Location -LiteralPath '{ROOT}'
$env:PYTHONUTF8='1'
$recPython=Join-Path (Get-Location) '.venv_repro/Scripts/python.exe'
& $recPython -m scripts.reproduce_stage --mode one --run-id ('resume_lr_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
```

若重新建环境（仅Windows本机、PowerShell、项目根）：系统Python3.12 `-m venv .venv_repro_new`，再用新环境解释器 `-m pip install -r requirements-stage-lock.txt`；不覆盖现有环境。实际完整命令与Linux区别见AUTODL_GUIDE。现有项目环境已就绪，不需要重复安装。

后续优先：按LEARNING_GUIDE第1—3节手写小练习；理解AUC与LogLoss差异；读same-structure消融。再预登记扩样/其他数据留出协议。本测试已观察，禁止把继续调参的同测试成绩包装为独立验证。

严格epoch断点续训尚未支持（无优化器/RNG状态checkpoint）；中断保留run，使用新run_id重跑，不覆盖或删除失败证据。结束前先看status.json；若状态failed，从train.log追溯。
''')
    # Index every deliverable file without enumerating virtual environments.
    rows=[]
    for folder in ['rec_lab','scripts','tests','configs','data','verification','runs','docs','student_work','exercises','learning','migration']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=str(p.relative_to(ROOT))
            if folder in ['student_work','exercises','learning']: purpose='用户练习/历史记录，保留'; rebuild='否，勿覆盖'; owner='用户'
            elif folder=='data' and 'raw' in p.parts:purpose='原始数据/来源';rebuild='官方归档可重下；student_phase1保留';owner='上游/下载脚本'
            elif folder=='data':purpose='固定切分/审计/逐行标识';rebuild='是，prepare_stage（新版本路径）';owner='prepare_stage'
            elif folder=='runs':purpose='实验配置/权重/日志/预测/指标/源码快照';rebuild='是，reproduce_stage（新run_id）';owner='实验脚本'
            elif folder=='verification':purpose='验证与修改前备份';rebuild='日志可重跑；原版备份勿覆盖';owner='维护者'
            elif folder=='docs':purpose='学习/交接/研究报告';rebuild='报告可脚本生成；解释需人工维护';owner='维护者'
            else:purpose='源码/配置/测试/迁移';rebuild='源码保留；迁移包可重建';owner='维护者'
            rows.append([str(p),rel,purpose,rebuild,owner])
    with (ROOT/'docs/FILE_INVENTORY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['absolute_path','relative_path','purpose_and_generation','rebuildable','maintainer']);writer.writerows(rows)
    key=[('rec_lab/data.py','特征和编码','源码，人工维护'),('rec_lab/models.py','LR以外模型结构','源码，人工维护'),('rec_lab/runner.py','训练/保存/评价','源码，人工维护'),
         ('scripts/prepare_stage.py','真实数据处理入口','源码，人工维护'),('scripts/reproduce_stage.py','冻结方案与一键实验','源码，人工维护'),
         ('scripts/verify_stage.py','真实保存重载验证','源码，人工维护'),('requirements-stage-lock.txt','实际依赖','pip freeze，可重建'),
         ('data/raw/KuaiRand-Pure.tar.gz','官方压缩包','fetch_kuairand，可重下'),('data/raw/provenance_stage.json','原文件大小/哈希/用途','build_stage_docs，可重建'),
         (DATA+'/manifest.json','数据协议/精确边界/分布','prepare_stage，可重建'),(DATA+'/split_membership.csv','每行所属集合','prepare_stage，可重建'),
         *[(DATA+'/'+s+'.csv',s+'固定数据','prepare_stage，可重建') for s in ['train','valid','test']],
         ('runs/'+RUN+'/protocol.json','实验前假设和参数','reproduce_stage，可新run重建'),('runs/'+RUN+'/selection.json','测试前冻结选择','reproduce_stage，可新run重建'),
         ('runs/'+RUN+'/summary.csv','3种子均值/SD','reproduce_stage，可重建'),('runs/'+RUN+'/comparison.csv','所有逐次指标','reproduce_stage，可重建'),
         ('runs/'+RUN+'/group_comparison.csv','逐次分组指标','build_stage_docs，可重建'),('runs/'+RUN+'/test_metrics.png','指标图','reproduce_stage，可重建'),
         ('runs/'+RUN+'/environment.json','实测环境','reproduce_stage，历史快照保留'),('verification/full_stage_20261107.log','正式运行日志','PowerShell重定向，历史保留'),
         ('student_work/01_inspect_data.py','你的练习入口','用户维护，不覆盖'),('learning/math','你的数学练习目录','用户维护，不覆盖'),
         ('migration/video_rec_stage_handoff.zip','最终代码文档迁移包','package_stage --out migration/video_rec_stage_handoff.zip，可重建')]
    for run in run_dirs:
        for name in ['config.json','encoder.json','metrics.json','valid_predictions.csv','test_predictions.csv','history.csv','train.log','resources.json']:
            key.append((str((run/name).relative_to(ROOT)),'对应run '+name,'reproduce_stage，新run可重建；维护者负责解释'))
        for name in ['model.pt','model.pkl','constant.json']:
            if (run/name).exists():key.append((str((run/name).relative_to(ROOT)),'模型/概率状态','reproduce_stage，新run可重建'))
    write('FILE_MAP.md',f'''
# 实际文件地图

唯一项目根：`{ROOT}`。所有表格路径是本机真实位置；不存在另一套完整项目。外层旧环境保持原状，新环境在根下.venv_repro。聊天工作区staging只是编辑交接暂存，后续工作以本项目为准。

下表逐项覆盖主要文件、模型、预测、日志；完整清单见 [FILE_INVENTORY.csv](FILE_INVENTORY.csv)，每行含绝对路径、相对路径、用途/生成方式、可重建性和维护者。源码及文档由项目维护者维护；原始公开数据由上游维护；脚本输出可用新路径重建；student_work/exercises/learning由你维护。

{table(['实际绝对路径','项目相对路径','用途','生成方式 / 重建 / 维护'],[[str(ROOT/rel),rel,purpose,how] for rel,purpose,how in key])}

每run还有data_manifest.json、versions.json、status.json；套件有requirements-lock.txt、source_hashes.json和source_snapshot。没有Git版本时用该快照与哈希定位实际实现。旧examples不纳入本轮汇总，也没有覆盖。
''')
    print('Documentation and provenance generated:',ROOT/'docs')

if __name__=='__main__':main()
