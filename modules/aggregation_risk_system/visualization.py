# -*- coding: utf-8 -*-
"""
결과 시각화.

classify_risks()로 최종 등급까지 매겨진 DataFrame을 받아 4종의 그래프를
out_dir에 PNG로 저장합니다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLOR_MAP = {"안심": "#2ca02c", "중간": "#1f77b4",
             "위험": "#ff7f0e", "고위험": "#d62728"}


def visualize(risk_df: pd.DataFrame, out_dir: Path) -> None:
    # 1. 정확도 분포 (위험도 색상)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for level in ["안심", "중간", "위험", "고위험"]:
        sub = risk_df[risk_df['Final_Risk'] == level]
        ax.scatter(sub.index, sub['Mean_Accuracy'],
                   label=level, color=COLOR_MAP[level], alpha=0.55, s=15)
    smoothed = risk_df['Mean_Accuracy'].rolling(20, center=True, min_periods=1).mean()
    ax.plot(risk_df.index, smoothed, color='black', lw=2, label='Smoothed Mean Acc')
    ax.axhline(y=0.5, color='gray', ls='--', lw=1, alpha=0.7, label='Baseline (0.5)')
    ax.set(xlabel='조합 Index', ylabel='Mean Accuracy',
           title='통합 위험도 기반 정확도 분포')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'risk_accuracy_distribution.png', dpi=200)
    plt.close()

    # 2. 위험도 분포 파이차트
    dist = risk_df['Final_Risk'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    colors = [COLOR_MAP.get(l, '#7f7f7f') for l in dist.index]
    ax.pie(dist.values, labels=dist.index, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.set_title('최종 위험도 분포')
    plt.savefig(out_dir / 'risk_distribution_pie.png', dpi=200)
    plt.close()

    # 3. 상위 위험 조합 Top 20
    top20 = (risk_df[risk_df['Final_Risk'].isin(["고위험", "위험"])]
             .sort_values('Mean_Accuracy', ascending=False)
             .head(20))
    if not top20.empty:
        plt.figure(figsize=(12, 7), dpi=150)
        bar_colors = [COLOR_MAP.get(l, '#7f7f7f') for l in top20['Final_Risk']]
        plt.barh(range(len(top20)), top20['Mean_Accuracy'], color=bar_colors)
        plt.yticks(range(len(top20)),
                   [c[:50] for c in top20['Feature_Combination']], fontsize=7)
        plt.xlabel('Mean Accuracy')
        plt.title('고위험·위험 Top 20 특성 조합')
        plt.tight_layout()
        plt.savefig(out_dir / 'top20_high_risk_combinations.png', dpi=200)
        plt.close()

    # 4. 모델별 정확도 박스플롯
    model_acc_cols = [c for c in risk_df.columns if c.endswith('_Accuracy')
                      and not c.startswith('Mean')]
    if model_acc_cols:
        melt = risk_df[model_acc_cols].melt(var_name='Model', value_name='Accuracy')
        melt['Model'] = melt['Model'].str.replace('_Accuracy', '')
        plt.figure(figsize=(12, 5), dpi=150)
        sns.boxplot(data=melt, x='Model', y='Accuracy')
        plt.xticks(rotation=30, ha='right')
        plt.title('모델별 조합 정확도 분포')
        plt.tight_layout()
        plt.savefig(out_dir / 'model_accuracy_boxplot.png', dpi=200)
        plt.close()
