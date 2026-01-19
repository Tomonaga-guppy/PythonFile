"""
ViTPoseとMocapを比較
1. 被験者ごとにViTPose結果とMocap結果の比較プロット
    比較内容
    - 関節角度, 歩行速度, 遊脚期時間のSI, 歩行周期, ストライド長
2. 全被験者結果の集計
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def plot_angles_compare(vit_angle_df, mocap_angle_df,vit_gait_cycles_r, vit_gait_cycles_l, output_dir):
    """
    関節角度csvを読み込み、Mocap結果と比較プロット
    """
    # 比較フレーム範囲の決定 mocapデータを参照(vitのフレームを参考にmocap側の使用フレームを決定している)
    compare_frame_range = mocap_angle_df.index

    # 股関節の屈曲伸展プロット
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title("Hip Flexion/Extension Comparison")
    axs.set_xlabel("Frame [-]")
    axs.set_ylabel("Flexion angle [deg]")
    plt.plot(compare_frame_range, vit_angle_df['R_Hip_FlEx'].loc[compare_frame_range], label='ViT R', color='tab:orange')
    plt.plot(compare_frame_range, vit_angle_df['L_Hip_FlEx'].loc[compare_frame_range], label='ViT L', color='tab:blue')
    plt.plot(compare_frame_range, mocap_angle_df['R_Hip_FlEx'].loc[compare_frame_range], label='Mocap R', color='tab:orange', linestyle='dashed')
    plt.plot(compare_frame_range, mocap_angle_df['L_Hip_FlEx'].loc[compare_frame_range], label='Mocap L', color='tab:blue', linestyle='dashed')
    for gc in vit_gait_cycles_r:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:orange', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_r[-1][-1]-0.5, vit_gait_cycles_r[-1][-1]+0.5, color='tab:orange', linestyle='dashed', alpha=0.3, label='IC R') # 最後の歩行周期終端
    for gc in vit_gait_cycles_l:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:blue', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_l[-1][-1]-0.5, vit_gait_cycles_l[-1][-1]+0.5, color='tab:blue', linestyle='dashed', alpha=0.3, label='IC L') # 最後の歩行周期終端
    axs.legend()
    plt.savefig(output_dir / "Hip_Flexion_Extension_Comparison.png")
    plt.close()
    
    # 膝関節の屈曲伸展プロット
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title("Knee Flexion/Extension Comparison")
    axs.set_xlabel("Frame [-]")
    axs.set_ylabel("Flexion angle [deg]")
    plt.plot(compare_frame_range, vit_angle_df['R_Knee_FlEx'].loc[compare_frame_range], label='ViT R', color='tab:orange')
    plt.plot(compare_frame_range, vit_angle_df['L_Knee_FlEx'].loc[compare_frame_range], label='ViT L', color='tab:blue')
    plt.plot(compare_frame_range, mocap_angle_df['R_Knee_FlEx'].loc[compare_frame_range], label='Mocap R', color='tab:orange', linestyle='dashed')
    plt.plot(compare_frame_range, mocap_angle_df['L_Knee_FlEx'].loc[compare_frame_range], label='Mocap L', color='tab:blue', linestyle='dashed')
    for gc in vit_gait_cycles_r:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:orange', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_r[-1][-1]-0.5, vit_gait_cycles_r[-1][-1]+0.5, color='tab:orange', linestyle='dashed', alpha=0.3, label='IC R') # 最後の歩行周期終端
    for gc in vit_gait_cycles_l:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:blue', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_l[-1][-1]-0.5, vit_gait_cycles_l[-1][-1]+0.5, color='tab:blue', linestyle='dashed', alpha=0.3, label='IC L') # 最後の歩行周期終端
    axs.legend()
    plt.savefig(output_dir / "Knee_Flexion_Extension_Comparison.png")
    plt.close()
    
    # 足関節の底背屈プロット
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title("Ankle Plantar/Dorsiflexion Comparison")
    axs.set_xlabel("Frame [-]")
    axs.set_ylabel("Dorsiflexion angle [deg]")
    plt.plot(compare_frame_range, vit_angle_df['R_Ankle_PlDo'].loc[compare_frame_range], label='ViT R', color='tab:orange')
    plt.plot(compare_frame_range, vit_angle_df['L_Ankle_PlDo'].loc[compare_frame_range], label='ViT L', color='tab:blue')
    plt.plot(compare_frame_range, mocap_angle_df['R_Ankle_PlDo'].loc[compare_frame_range], label='Mocap R', color='tab:orange', linestyle='dashed')
    plt.plot(compare_frame_range, mocap_angle_df['L_Ankle_PlDo'].loc[compare_frame_range], label='Mocap L', color='tab:blue', linestyle='dashed')
    for gc in vit_gait_cycles_r:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:orange', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_r[-1][-1]-0.5, vit_gait_cycles_r[-1][-1]+0.5, color='tab:orange', linestyle='dashed', alpha=0.3, label='IC R') # 最後の歩行周期終端
    for gc in vit_gait_cycles_l:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:blue', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_l[-1][-1]-0.5, vit_gait_cycles_l[-1][-1]+0.5, color='tab:blue', linestyle='dashed', alpha=0.3, label='IC L') # 最後の歩行周期終端
    axs.legend()
    plt.savefig(output_dir / "Ankle_Plantar_Dorsiflexion_Comparison.png")
    plt.close()
    
    # 股関節の外転内転プロット
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title("Hip Adduction/Abduction Comparison")
    axs.set_xlabel("Frame [-]")
    axs.set_ylabel("Adduction/Abduction angle [deg]")
    plt.plot(compare_frame_range, vit_angle_df['R_Hip_AdAb'].loc[compare_frame_range], label='ViT R', color='tab:orange')
    plt.plot(compare_frame_range, vit_angle_df['L_Hip_AdAb'].loc[compare_frame_range], label='ViT L', color='tab:blue')
    plt.plot(compare_frame_range, mocap_angle_df['R_Hip_AdAb'].loc[compare_frame_range], label='Mocap R', color='tab:orange', linestyle='dashed')
    plt.plot(compare_frame_range, mocap_angle_df['L_Hip_AdAb'].loc[compare_frame_range], label='Mocap L', color='tab:blue', linestyle='dashed')
    for gc in vit_gait_cycles_r:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:orange', linestyle='dotted', alpha=0.5) 
    axs.axvspan(vit_gait_cycles_r[-1][-1]-0.5, vit_gait_cycles_r[-1][-1]+0.5, color='tab:orange', linestyle='dashed', alpha=0.3, label='IC R') # 最後の歩行周期終端
    for gc in vit_gait_cycles_l:
        axs.axvspan(gc[0]-0.5, gc[0]+0.5, color='tab:blue', linestyle='dotted', alpha=0.5)
    axs.axvspan(vit_gait_cycles_l[-1][-1]-0.5, vit_gait_cycles_l[-1][-1]+0.5, color='tab:blue', linestyle='dashed', alpha=0.3, label='IC L') # 最後の歩行周期終端
    axs.legend()
    plt.savefig(output_dir / "Hip_Adduction_Abduction_Comparison.png")
    plt.close()

def calculate_and_save_angle_mae(vit_angle_df, mocap_angle_df, vit_gait_cycles_r, vit_gait_cycles_l, output_dir):
    """
    関節角度のMAE算出
    """
    
    for side, gait_cycles in zip(['R', 'L'], [vit_gait_cycles_r, vit_gait_cycles_l]):
        mae_hip_flex = []
        mae_knee_flex = []
        mae_ankle_pldo = []
        mae_hip_adab = []
        
        for gc in gait_cycles:
            start_frame = gc[0]
            end_frame = gc[-1]
            vit_segment = vit_angle_df.loc[start_frame:end_frame]
            mocap_segment = mocap_angle_df.loc[start_frame:end_frame]
            
            mae_hip_flex.append(np.mean(np.abs(vit_segment[f'{side}_Hip_FlEx'] - mocap_segment[f'{side}_Hip_FlEx'])))
            mae_knee_flex.append(np.mean(np.abs(vit_segment[f'{side}_Knee_FlEx'] - mocap_segment[f'{side}_Knee_FlEx'])))
            mae_ankle_pldo.append(np.mean(np.abs(vit_segment[f'{side}_Ankle_PlDo'] - mocap_segment[f'{side}_Ankle_PlDo'])))
            mae_hip_adab.append(np.mean(np.abs(vit_segment[f'{side}_Hip_AdAb'] - mocap_segment[f'{side}_Hip_AdAb'])))
        
        # 結果の保存
        mae_df = pd.DataFrame({
            'Gait_Cycle_Start_Frame': [gc[0] for gc in gait_cycles],
            'MAE_Hip_Flexion_Extension': mae_hip_flex,
            'MAE_Knee_Flexion_Extension': mae_knee_flex,
            'MAE_Ankle_Plantar_Dorsiflexion': mae_ankle_pldo,
            'MAE_Hip_Adduction_Abduction': mae_hip_adab
        })
        
        # 全体の平均MAEもCSVに追加
        mean_mae = {
            'Gait_Cycle_Start_Frame': ['Mean'],
            'MAE_Hip_Flexion_Extension': [np.mean(mae_hip_flex)],
            'MAE_Knee_Flexion_Extension': [np.mean(mae_knee_flex)],
            'MAE_Ankle_Plantar_Dorsiflexion': [np.mean(mae_ankle_pldo)],
            'MAE_Hip_Adduction_Abduction': [np.mean(mae_hip_adab)]
        }
        mae_df = pd.concat([mae_df, pd.DataFrame(mean_mae)], ignore_index=True)
        mae_df.to_csv(output_dir / f'MAE_Angle_{side}.csv', index=False)

def calculate_and_save_gaitparam_mae(vit_gaitparam_df, mocap_gaitparam_df, output_dir):
    """
    歩行パラメータのMAEを算出して保存
    対象パラメータ： 歩行速度, ストライド時間, 遊脚期時間, 歩隔
    vitのic, toフレームに合わせてmocapも計算しているため, ストライド時間と遊脚期時間はMAEが必ず0になるので意味がないが、一応計算して保存しておく
    """
    mae_results = {}
    for column in ['gait_speed', 'stride_time', 'swing_duration', 'stride_width']:
        if column in vit_gaitparam_df.columns and column in mocap_gaitparam_df.columns:
            vit_values = vit_gaitparam_df[column].values
            mocap_values = mocap_gaitparam_df[column].values
            mae = np.mean(np.abs(vit_values - mocap_values))
            mae_results[f'MAE_{column}'] = mae
    
    # 結果の保存
    mae_df = pd.DataFrame([mae_results])
    # print(f"Gait Parameters MAE:\n{mae_df}")
    mae_df.to_csv(output_dir / 'MAE_Gait_Parameters.csv', index=False)
    
def process_pairs(vit_dir, mocap_dir):
    """
    関節角度と歩行パラメータの比較・MAEの算出
    1. 関節角度csvを読み込み、Mocap結果と比較プロット
    2. 歩行パラメータcsvを読み込み、Mocap結果と比較・MAE算出
    3. 結果を表示・保存
    """
    
    # 結果の出力用ディレクトリ作成
    output_dir = vit_dir / "compare_mocap_results"
    output_dir.mkdir(exist_ok=True)
    
    # ViTでの歩行周期の読み取り
    vit_gait_cycles_npz = next(vit_dir.glob("gait_cycles_*.npz"), None)
    vit_gait_cycles = np.load(vit_gait_cycles_npz, allow_pickle=True)
    # print(f"vit_gait_cycles: {vit_gait_cycles}")
    vit_gait_cycles_r = vit_gait_cycles['gait_cycle_r']
    vit_gait_cycles_l = vit_gait_cycles['gait_cycle_l']
    
    # 1.関節角度の比較 ===================================================
    mocap_angle_csv = next(mocap_dir.glob("angle_60Hz_*.csv"), None)
    vit_angle_csv = next(vit_dir.glob("angles.csv"), None)
    if mocap_angle_csv is None:
        print(f"[SKIP] missing mocap angle csv in {mocap_dir}")
        return
    if vit_angle_csv is None:
        print(f"[SKIP] missing ViT angle csv in {vit_dir}")
        return
    mocap_angle_df = pd.read_csv(mocap_angle_csv, index_col=0)
    vit_angle_df = pd.read_csv(vit_angle_csv, index_col=0)
    
    # 関節角度プロット比較
    plot_angles_compare(vit_angle_df, mocap_angle_df, vit_gait_cycles_r, vit_gait_cycles_l, output_dir)
    # MAEの算出
    calculate_and_save_angle_mae(vit_angle_df, mocap_angle_df, vit_gait_cycles_r, vit_gait_cycles_l, output_dir)
    
    # 2.歩行パラメータの比較 ===================================================
    mocap_gaitparam_csv = next(mocap_dir.glob("gait_parameters_*.csv"), None)
    vit_gaitparam_csv = next(vit_dir.glob("gait_parameters_*.csv"), None)
    if mocap_gaitparam_csv is None:
        print(f"[SKIP] missing mocap gait param csv in {mocap_dir}")
        return
    if vit_gaitparam_csv is None:
        print(f"[SKIP] missing ViT gait param csv in {vit_dir}")
        return
    mocap_gaitparam_df = pd.read_csv(mocap_gaitparam_csv)
    vit_gaitparam_df = pd.read_csv(vit_gaitparam_csv)
    
    calculate_and_save_gaitparam_mae(vit_gaitparam_df, mocap_gaitparam_df, output_dir)
    

def main():
    root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")

    # =====================================================================
    # 1. 各被験者でMAEを算出
    # =====================================================================
    print("=== Start comparison of ViTPose and Mocap results ===")
    print("Step 1: Process each subject =================================")
    subjects_processed = []

    for sub_i in range(1, 11):
        # # 被験者の対象しぼるならここで指定 不要ならコメントアウト
        # if sub_i != 1:
        #     print(f"[SKIP] not check {sub_i} now")
        #     continue
        sub_dir = root_dir / f"sub{sub_i}"
        if not sub_dir.exists():
            print(f"[SKIP] missing: {sub_dir}")
            continue

        thera_dir = sub_dir / f"thera{sub_i}-0"
        
        mocap_dir = thera_dir / "mocap"
        if not mocap_dir.exists():
            print(f"[SKIP] missing: {mocap_dir}")
            continue
        
        ViT_dir = thera_dir / "ViTPose_Results"
        if not ViT_dir.exists():
            print(f"[SKIP] missing: {ViT_dir}")
            continue
        
        print(f"[PROCESS] sub{sub_i} thera{sub_i}-0")
        process_pairs(ViT_dir, mocap_dir)
        subjects_processed.append(sub_i)

    # =====================================================================
    # 2. 全被験者結果の集計
    # =====================================================================
    print("\nStep 2: Aggregate results across subjects ====================")
    out_dir = root_dir / "compare_mocap_results_ALL"
    out_dir.mkdir(exist_ok=True)

    def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
        """列ごとに mean, sd, median, q1, q3, n を返す．"""
        rows = []
        for col in df.columns:
            x = pd.to_numeric(df[col], errors='coerce')
            n = int(x.notna().sum())
            rows.append({
                "metric": col,
                "mean": float(x.mean()) if n > 0 else np.nan,
                "sd": float(x.std(ddof=1)) if n >= 2 else np.nan,
                "median": float(x.median()) if n > 0 else np.nan,
                "q1": float(x.quantile(0.25)) if n > 0 else np.nan,
                "q3": float(x.quantile(0.75)) if n > 0 else np.nan,
                "n": n,
            })
        return pd.DataFrame(rows)

    # 2-1) 歩行パラメータMAE（被験者単位）
    gait_rows = []
    for sub_i in subjects_processed:
        thera_dir = root_dir / f"sub{sub_i}" / f"thera{sub_i}-0"
        vit_dir = thera_dir / "ViTPose_Results"
        res_dir = vit_dir / "compare_mocap_results"
        mae_csv = res_dir / "MAE_Gait_Parameters.csv"
        if not mae_csv.exists():
            continue
        df = pd.read_csv(mae_csv)
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        row = {k: pd.to_numeric(v, errors='coerce') for k, v in row.items()}
        row["subject"] = f"sub{sub_i}"
        gait_rows.append(row)

    if gait_rows:
        gait_sub_df = pd.DataFrame(gait_rows).set_index("subject")
        gait_sub_df.to_csv(out_dir / "subject_level_gaitparam_mae.csv")
        summary_stats(gait_sub_df).to_csv(out_dir / "summary_gaitparam_mae.csv", index=False)
    else:
        print("[INFO] MAE_Gait_Parameters.csv が見つからなかったため，歩行パラメータの全体集計はスキップします")

    # 2-2) 角度MAE（存在する場合のみ）
    angle_rows = []
    pooled_cycle_rows = []

    def _read_angle_mae_csv(path: Path):
        """MAE_Angle_*.csv から Mean 行と各周期行（Mean除外）を返す．"""
        df = pd.read_csv(path)
        if "Gait_Cycle_Start_Frame" not in df.columns:
            return None, None
        # Mean 行
        mean_mask = df["Gait_Cycle_Start_Frame"].astype(str).str.lower() == "mean"
        mean_row = df.loc[mean_mask].iloc[0] if mean_mask.any() else df.iloc[-1]
        # 周期行（Mean除外）
        cyc_df = df.loc[~mean_mask].copy() if mean_mask.any() else df.iloc[:-1].copy()
        return mean_row, cyc_df

    for sub_i in subjects_processed:
        thera_dir = root_dir / f"sub{sub_i}" / f"thera{sub_i}-0"
        vit_dir = thera_dir / "ViTPose_Results"
        res_dir = vit_dir / "compare_mocap_results"
        fR = res_dir / "MAE_Angle_R.csv"
        fL = res_dir / "MAE_Angle_L.csv"
        if (not fR.exists()) and (not fL.exists()):
            continue

        row = {"subject": f"sub{sub_i}"}

        if fR.exists():
            mean_row, cyc_df = _read_angle_mae_csv(fR)
            if mean_row is not None:
                for c in mean_row.index:
                    if str(c).startswith("MAE_"):
                        row[f"R_{c.replace('MAE_', '')}"] = pd.to_numeric(mean_row[c], errors='coerce')
            if cyc_df is not None and len(cyc_df) > 0:
                tmp = cyc_df[[c for c in cyc_df.columns if str(c).startswith("MAE_")]].copy()
                tmp.columns = [f"R_{c.replace('MAE_', '')}" for c in tmp.columns]
                tmp["subject"] = f"sub{sub_i}"
                pooled_cycle_rows.append(tmp)

        if fL.exists():
            mean_row, cyc_df = _read_angle_mae_csv(fL)
            if mean_row is not None:
                for c in mean_row.index:
                    if str(c).startswith("MAE_"):
                        row[f"L_{c.replace('MAE_', '')}"] = pd.to_numeric(mean_row[c], errors='coerce')
            if cyc_df is not None and len(cyc_df) > 0:
                tmp = cyc_df[[c for c in cyc_df.columns if str(c).startswith("MAE_")]].copy()
                tmp.columns = [f"L_{c.replace('MAE_', '')}" for c in tmp.columns]
                tmp["subject"] = f"sub{sub_i}"
                pooled_cycle_rows.append(tmp)

        # R/L の平均（両方ある場合のみ）
        for base in [
            "Hip_Flexion_Extension",
            "Knee_Flexion_Extension",
            "Ankle_Plantar_Dorsiflexion",
            "Hip_Adduction_Abduction",
        ]:
            rkey = f"R_{base}"
            lkey = f"L_{base}"
            if rkey in row and lkey in row:
                row[f"MeanRL_{base}"] = float(np.nanmean([row[rkey], row[lkey]]))

        angle_rows.append(row)

    if angle_rows:
        ang_sub_df = pd.DataFrame(angle_rows).set_index("subject")
        ang_sub_df.to_csv(out_dir / "subject_level_angle_mae.csv")
        summary_stats(ang_sub_df).to_csv(out_dir / "summary_angle_mae.csv", index=False)

        # 参考: 全周期 pooled 平均（被験者・周期数に重みがつく）
        if pooled_cycle_rows:
            pooled_df = pd.concat(pooled_cycle_rows, ignore_index=True)
            num_cols = [c for c in pooled_df.columns if c != "subject"]
            pooled_mean = pooled_df[num_cols].mean(numeric_only=True)
            pooled_out = pooled_mean.to_frame(name="pooled_cycle_mean").reset_index().rename(columns={"index": "metric"})
            pooled_out.to_csv(out_dir / "pooled_cycle_angle_mae.csv", index=False)
    else:
        print("[INFO] MAE_Angle_R/L.csv が見つからなかったため，角度MAEの全体集計はスキップします")

    print(f"[DONE] 全体集計の出力先: {out_dir}")
    
if __name__ == "__main__":
    main()