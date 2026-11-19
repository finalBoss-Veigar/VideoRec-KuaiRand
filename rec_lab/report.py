"""Self-contained experiment report with curves, calibration and seed summaries."""
import base64
import html
import io
from pathlib import Path
import pandas as pd
from .runner import compare,read_json
from .experiments import seed_summary

def figure_data(fig):
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()

def report(run_dirs,out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out_dir)
    if out.exists():raise ValueError('Report directory exists; choose a new report version.')
    out.mkdir(parents=True)
    frame=compare(run_dirs,out/'comparison.csv');summary=seed_summary(frame)
    summary.to_csv(out/'seed_summary.csv',index=False)
    first=Path(run_dirs[0]);manifest=read_json(first/'data_manifest.json')
    kind=manifest['dataset_kind'];figures=[]
    for root in map(Path,run_dirs):
        history=pd.read_csv(root/'history.csv')
        if 'train_loss' not in history:continue
        fig,ax=plt.subplots(figsize=(6.6,3))
        ax.plot(history.epoch,history.train_loss,label='train');ax.plot(history.epoch,history.valid_loss,label='validation')
        ax.set(xlabel='Epoch',ylabel='Weighted BCE',title=root.name);ax.legend();ax.grid(alpha=.2)
        figures.append((root.name,figure_data(fig)));plt.close(fig)
    calibration=[]
    for root in map(Path,run_dirs):
        metrics=read_json(root/'metrics.json')
        for target,values in metrics['valid'].items():
            bins=values['calibration']
            if not bins:continue
            fig,ax=plt.subplots(figsize=(4.8,3.4))
            ax.plot([0,1],[0,1],'--',color='gray',label='ideal')
            ax.plot([x['mean_prediction'] for x in bins],[x['positive_rate'] for x in bins],'o-',label='observed')
            ax.set(xlabel='Mean predicted probability',ylabel='Observed positive rate',title=f'{root.name} / {target}',xlim=(0,1),ylim=(0,1));ax.legend();ax.grid(alpha=.2)
            calibration.append((f'{root.name} / {target}',figure_data(fig)));plt.close(fig)
    group_rows=[]
    for root in map(Path,run_dirs):
        for split in ['valid','test']:
            file=root/f'{split}_groups.json'
            if not file.exists():continue
            for target,groups in read_json(file).items():
                for group,values in groups.items():
                    group_rows.append({'run':root.name,'split':split,'target':target,'group':group,
                        **{k:v for k,v in values.items() if k not in ['calibration','auc_note']}})
    groups=pd.DataFrame(group_rows);groups.to_csv(out/'groups.csv',index=False)
    display=['run','split','target','auc','logloss','pr_auc_average_precision','brier_score','parameters','training_seconds']
    table=frame[display].to_html(index=False,float_format=lambda x:f'{x:.5f}',na_rep='undefined',escape=True)
    seed_columns=['model','task','features','like_weight','split','target','auc_count','auc_mean','auc_std','logloss_mean','logloss_std']
    summary_table=summary[seed_columns].to_html(index=False,float_format=lambda x:f'{x:.5f}',na_rep='undefined',escape=True)
    def imgs(items):return ''.join('<figure><img alt="'+html.escape(t)+'" src="'+u+'"><figcaption>'+html.escape(t)+'</figcaption></figure>' for t,u in items)
    scene=manifest.get('scene_tab');note=manifest.get('signal_note','')
    intro=f'数据类型 {kind}；场景tab {scene}；曝光策略 {manifest["policy"]}；数据身份 {manifest["data_id"]}。'
    limitations=('合成数据仅验证程序流程，不能作为真实推荐效果。' if kind=='synthetic' else
                 '当前结果是所选场景和样本上的离线实验，不能直接证明线上提升。')
    body=f'''<h1>视频反馈预测实验报告</h1><p>{html.escape(intro)}</p><p>{html.escape(note)}</p><p>{limitations}</p>
<h2>数据与评价口径</h2><p>训练和验证按全局时间划分。测试结果只在明确执行最终评价后出现。AUC无定义时标为undefined。PR指标使用Average Precision。活跃度按训练期记录数分组。</p>
{pd.DataFrame(manifest['split_audit']).T[['rows','users','videos','feedback_rate','like_rate']].to_html(escape=True)}
<h2>逐运行结果</h2><p>同一份数据和协议才允许合并；不同任务单独比较。训练时间不包含数据准备与最终报告生成。</p><div class="wide">{table}</div>
<h2>不同随机种子的汇总</h2><p>只有除种子外设置相同的运行才汇总。count是AUC可计算的运行数；只有一次运行时标准差没有定义。多种子波动不能代替总体置信区间。</p><div class="wide">{summary_table}</div>
<h2>训练与验证曲线</h2><p>不同任务权重下损失的量纲不同，不能只比较加权损失数值来选择权重。</p>{imgs(figures)}
<h2>验证集概率校准</h2><p>曲线接近对角线表示分箱预测概率与观测比例较接近；小样本分箱可能波动。分箱计数保存在各运行的metrics.json。</p>{imgs(calibration)}
<h2>分组评估</h2><p>完整分组数据见groups.csv。已见与未见用户及视频、训练期活跃度可分别检查。组样本少时不做强结论。</p>
<h2>你需要亲手写的结论</h2><ol><li>哪个基准最值得保留，理由是什么。</li><li>多任务对两个目标分别有什么影响。</li><li>变化是否大于种子波动，是否付出更多参数与计算成本。</li><li>有哪些失败案例、数据偏差和下一步实验。</li></ol>'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>视频反馈预测实验报告</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1080px;margin:32px auto;padding:0 20px;color:#20302c}h1,h2{color:#142e27}table{border-collapse:collapse;font-size:13px;margin:20px 0}td,th{border:1px solid #d5ded9;padding:9px;text-align:left}th{background:#eaf2ee}.wide{overflow:auto}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#56675e}p{overflow-wrap:anywhere}@media print{.wide{overflow:visible}table{font-size:9px}figure{break-inside:avoid}}</style><body>'+body+'</body></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    (out/'analysis_template.md').write_text('# 本轮实验复盘\n\n'+intro+'\n\n'+limitations+'\n\n## 我的问题\n\n## 数据与标签\n\n## 验证集选择规则\n\n## 结果与成本\n\n## 退步和失败案例\n\n## 下一步\n',encoding='utf-8')
    return {'report':str(out/'report.html'),'comparison':str(out/'comparison.csv'),'seed_summary':str(out/'seed_summary.csv')}
