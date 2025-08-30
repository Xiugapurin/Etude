# --- file: run_master_evaluation.py (v4 - Robust Final) ---

import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# 假設您的計算器類別都位於 evaluation/ 目錄下
from evaluation import RGCCalculator, IPECalculator, WPDCalculator

def get_genre_from_dirname(dir_name: str) -> str:
    dir_name_upper = dir_name.upper()
    if "CPOP" in dir_name_upper: return "CPOP"
    elif "JPOP" in dir_name_upper: return "JPOP"
    elif "KPOP" in dir_name_upper: return "KPOP"
    elif "WESTERN" in dir_name_upper: return "WESTERN"
    else: return "UNKNOWN"


def run_full_evaluation():
    """
    主函式：計算 WPD, RGC, IPE 三個指標，並進行綜合分析與視覺化。
    【v4.0 - 已修復 KeyError 並增加穩健性】
    """
    EVAL_DIR = "./dataset/eval"
    METADATA_PATH = os.path.join(EVAL_DIR, "metadata.json")
    VERSIONS = ["cover", "etude_e", "etude_d", "etude_d_d", "picogen", "amtapc", "music2midi"]
    VERSION_DISPLAY_NAMES = {
        "cover": "Human", "etude_e": "Etude Extractor", "etude_d_d": "Etude Decoder - Default",
        "etude_d": "Etude Decoder - Prompted", "picogen": "PiCoGen", "amtapc": "AMT-APC", "music2midi": "Music2MIDI"
    }

    # 參數設定
    wpd_params = {'subsample_step': 1, 'trim_seconds': 10}
    rgc_params = {'top_k': 8}
    ipe_params = {'n_gram': 8, 'n_clusters': 16}

    print("Initializing calculators...")
    wpd_calc = WPDCalculator(**wpd_params)
    rgc_calc = RGCCalculator(**rgc_params)
    ipe_calc = IPECalculator(**ipe_params)
    
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if not metadata:
            print(f"❌ 警告：metadata.json 檔案為空，無法進行分析。")
            return
        print(f"✅ 成功讀取 metadata.json，將分析 {len(metadata)} 首歌曲。")
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到 metadata.json 檔案於 {METADATA_PATH}")
        return
        
    results_list = []
    for song_data in tqdm(metadata, desc="Master Analysis"):
        dir_name = song_data.get("dir_name")
        if not dir_name: continue
        
        song_dir = os.path.join(EVAL_DIR, dir_name)
        origin_wav_path = os.path.join(song_dir, "origin.wav")
        if not os.path.exists(origin_wav_path): continue
            
        for version in VERSIONS:
            wav_path = os.path.join(song_dir, f"{version}.wav")
            mid_path = os.path.join(song_dir, f"{version}.mid")
            
            if not os.path.exists(wav_path) and not os.path.exists(mid_path):
                continue

            result_row = {'song': dir_name, 'version': version}

            wpd_res = wpd_calc.calculate_wpd(origin_wav_path, wav_path, song_dir)
            if "error" not in wpd_res: result_row.update(wpd_res)

            rgc_res = rgc_calc.calculate_rgc(mid_path)
            if "error" not in rgc_res: result_row.update(rgc_res)
                
            ipe_res = ipe_calc.calculate_ipe(mid_path)
            if "error" not in ipe_res: result_row.update(ipe_res)
            
            # 確保至少有一個分數被成功計算
            if len(result_row) > 2:
                 results_list.append(result_row)

    # --- 4. 分析與呈現 ---
    df_all = pd.DataFrame(results_list)

    # --- 【關鍵修改 #2】在進行任何操作前，先檢查 DataFrame 是否為空 ---
    if df_all.empty:
        print("\n❌ 未能收集到任何有效的評分數據，無法生成報告和圖表。")
        print("   請檢查您的 eval 目錄結構、檔案名稱以及 metadata.json 的內容是否正確。")
        return
        
    df_all['display_name'] = df_all['version'].map(VERSION_DISPLAY_NAMES).fillna(df_all['version'])

    print("\n\n" + "="*20 + " 綜合評估報告 " + "="*20)
    
    existing_metrics = [col for col in ['wpd_score', 'rgc_score', 'ipe_score'] if col in df_all.columns]
    
    if not existing_metrics:
        print("所有指標均未能成功計算，無法生成報告。")
        return

    # 打印每個指標的詳細統計
    for metric in existing_metrics:
        print(f"\n--- 指標: {metric.upper()} 詳細統計 ---")
        summary = df_all.groupby('display_name')[metric].describe().sort_values('mean', ascending=False)
        print(summary)
            
    # 打印一個包含所有指標平均值的總表
    print("\n--- 所有指標平均分總覽 ---")
    mean_summary = df_all.groupby('display_name')[existing_metrics].mean()
    print(mean_summary.sort_values(existing_metrics[0], ascending=False))

    # --- 5. 視覺化 ---
    metrics_to_plot = {
        'WPD Score': 'wpd_score', 'RGC Score': 'rgc_score', 'IPE Score': 'ipe_score'
    }
    plotable_metrics = {title: col for title, col in metrics_to_plot.items() if col in df_all.columns}
    if not plotable_metrics:
        print("無數據可供繪圖。")
        return

    print("\n🎨 正在生成綜合評分分佈圖...")
    fig, axes = plt.subplots(len(plotable_metrics), 1, figsize=(12, 8 * len(plotable_metrics)), squeeze=False)
    fig.suptitle('Multi-Metric Evaluation', fontsize=20, y=0.98)
    for i, (title, metric_col) in enumerate(plotable_metrics.items()):
        ax = axes[i, 0]
        order = df_all.groupby('display_name')[metric_col].mean().sort_values(ascending=False).index
        sns.boxplot(data=df_all, x=metric_col, y='display_name', hue='display_name', order=order, palette='viridis', ax=ax, legend=False)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Score (Higher is Better)', fontsize=12)
        ax.set_ylabel('Version', fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_image_path = "master_evaluation_summary.png"
    plt.savefig(output_image_path, dpi=300)
    print(f"✅ 綜合評估圖表已成功儲存至: {output_image_path}")


if __name__ == '__main__':
    run_full_evaluation()