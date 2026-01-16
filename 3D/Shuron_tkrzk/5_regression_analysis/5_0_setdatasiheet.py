"""
回帰分析を行う前にPA,PTのデータシートを整えるスクリプト
患者や療法士の基本情報はデータシートから手動でまとめた
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk")
out_dir = ROOT_DIR / "regression"

pa_gait_params_list = []
pt_assi_params_list = []

for sub_dir in sorted(ROOT_DIR.glob("sub*")):
    for thera_dir in sorted(sub_dir.glob("thera*")):
        if thera_dir.name == "thera0":
            continue  # thera0は除外
        
        # PAの歩行パラメータデータシートを収集 ###################################################################################
        min_assi_pa_gait_params_path = sub_dir / "thera0" / "ViTPose_results" / "pa_gait_parameters.csv"
        pa_gait_params_path = thera_dir / "ViTPose_results" / "pa_gait_parameters.csv"
        
        min_assi_pa_gait_params = pd.read_csv(min_assi_pa_gait_params_path)
        pa_gait_params = pd.read_csv(pa_gait_params_path)
        
        # 介助による歩行速度の変化量を算出
        spped_delta = pa_gait_params.loc[0, "gait_speed"] - min_assi_pa_gait_params.loc[0, "gait_speed"]
        
        pa_gait_header = pa_gait_params.columns.tolist()
        # データシートに歩行速度変化量やpa_id, pt_idの列を追加
        if "gait_speed_delta" not in pa_gait_header:
            pa_gait_params["gait_speed_delta"] = spped_delta
        if "pt_id" not in pa_gait_header:
            pa_gait_params["pt_id"] = thera_dir.name.replace("thera", "")
        if "pa_id" not in pa_gait_header:
            pa_gait_params["pa_id"] = sub_dir.name.replace("sub", "")

        pa_gait_params_list.append(pa_gait_params)
        
        # PTの介助パラメータデータシートも収集 ###################################################################################
        pt_assi_params_path = thera_dir / "ViTPose_results" / "pt_assist_metrics.csv"
        pt_assi_params = pd.read_csv(pt_assi_params_path)
        
        # データシートにpa_id, pt_idの列を追加
        pt_assi_header = pt_assi_params.columns.tolist()
        if "pt_id" not in pt_assi_header:
            pt_assi_params["pt_id"] = thera_dir.name.replace("thera", "")
        if "pa_id" not in pt_assi_header:
            pt_assi_params["pa_id"] = sub_dir.name.replace("sub", "")
        pt_assi_params_list.append(pt_assi_params)

# PAデータの結合と保存 ###################################################################################
# リストを1つのデータフレームに結合
pa_gait_params_all = pd.concat(pa_gait_params_list, ignore_index=True)
pa_gait_params_all = pa_gait_params_all.apply(pd.to_numeric, errors='ignore')

# 列の順序を変更: pa_id, pt_id を先頭に、残りはそのまま
other_cols = [col for col in pa_gait_params_all.columns if col not in ['pa_id', 'pt_id', 'gait_speed', 'gait_speed_delta']]
pa_gait_params_all = pa_gait_params_all[['pa_id', 'pt_id', 'gait_speed', 'gait_speed_delta'] + other_cols]

# pa_id を若い順に並べ替え（pt_id も副次的にソート）
pa_gait_params_all = pa_gait_params_all.sort_values(by=['pa_id', 'pt_id', 'gait_speed', 'gait_speed_delta']).reset_index(drop=True)

print(f"Combined dataframe shape: {pa_gait_params_all.shape}")
print(pa_gait_params_all)

pa_gait_params_all.to_csv(out_dir / "pa_gait_parameters.csv", index=False)

# PTデータの結合と保存 ###################################################################################
# リストを1つのデータフレームに結合
pt_assi_params_all = pd.concat(pt_assi_params_list, ignore_index=True)
pt_assi_params_all = pt_assi_params_all.apply(pd.to_numeric, errors='ignore')

# 列の順序を変更: pa_id, pt_id を先頭に、残りはそのまま
other_cols = [col for col in pt_assi_params_all.columns if col not in ['pa_id', 'pt_id']]
pt_assi_params_all = pt_assi_params_all[['pa_id', 'pt_id'] + other_cols]
# pa_id を若い順に並べ替え（pt_id も副次的にソート）
pt_assi_params_all = pt_assi_params_all.sort_values(by=['pa_id', 'pt_id']).reset_index(drop=True)
print(f"Combined dataframe shape: {pt_assi_params_all.shape}")
print(pt_assi_params_all)
pt_assi_params_all.to_csv(out_dir / "pt_assist_parameters.csv", index=False)

"""
データまとめ結果例
pa_gait_parameters.csv
Combined dataframe shape: (10, 12)
   pa_id  pt_id  gait_speed  gait_speed_delta  symmetry_index_sw  stride_time  stride_width  hip_max_flex  hip_max_ext  knee_max_flex  ankle_max_pl  hip_max_ab
0      2      2    0.815331          0.196075          27.450980     1.383333      0.140979     31.021487   -14.657834      31.988449    -11.480403    6.991343
1      2      3    0.829941          0.210684          17.391304     1.266667      0.200703     31.682289   -14.025554      28.222876     -7.568479    6.583302
2      3      2    0.731531          0.076715          16.000000     1.083333      0.051501     19.773117   -11.547005      50.732860     -8.821265    6.233481
3      3      4    0.759765          0.104949         -14.634146     1.050000      0.118781     19.290911    -9.529688      55.907818     -2.596914    5.695646
4     15      1    0.691460          0.096041          19.607843     1.416667      0.126740     24.130400    -6.292372      50.867716     -8.380449    3.884971
5     15      2    0.645587          0.050168           3.773585     1.416667      0.116893     29.385691    -7.504816      61.572964     -4.437440    8.789557
6     15      4    0.625910          0.030491          25.000000     1.366667      0.143489     25.935527    -8.618433      55.965452     -9.137646    8.413019
7     16      1    0.673723          0.212121           4.651163     1.183333      0.097186     39.342413    -4.836024      61.765990     -6.056106    9.920073
8     16      7    0.540800          0.079197          20.000000     1.283333      0.115231     35.753722    -7.062754      69.179725     -0.280571   11.711558
9     16     15    0.636187          0.174585          -5.405405     1.050000      0.176543     30.637805    -3.175073      65.799358     -4.528609   11.559749

pt_assist_parameters.csv
Combined dataframe shape: (10, 7)
   pa_id  pt_id  hip_dist  hip_cc_x  hip_cc_y  hip_cc_z  hip_cc_lag
0      2      2  0.427177  0.997453  0.680783  0.898972    0.066667
1      2      3  0.551362  0.974443  0.435763  0.971202   -0.016667
2      3      2  0.410365  0.895645  0.309268  0.936150    0.033333
3      3      4  0.507203  0.958218  0.585975  0.999228    0.000000
4     15      1  0.470255  0.966347  0.715872  0.950461    0.033333
5     15      2  0.417440  0.926546  0.831813  0.905004    0.066667
6     15      4  0.363042  0.986001  0.388844  0.949537    0.033333
7     16      1  0.533195  0.903828  0.428500  0.791592    0.116667
8     16      7  0.396158  0.907818  0.489773  0.861180    0.083333
9     16     15  0.260234  0.979030  0.889150  0.999612    0.000000

pa_basic_data.csv(基本情報を手動でまとめたもの)
pa_id	pa_sex	pa_age	pa_height	pa_weight	pa_foot_len	knee_lat_to_malleolus	fac	brs_lower	sias_m_hip	sias_m_knee	sias_m_ankle	sias_m_total	sias_m_sens_sole	sias_m_prop_toe	mi_hip	mi_knee	mi_ankle	mi_total	days_post_onset	fim_walk	fim_motor	fim_cog	mmse
2	1	66	175	69.7	27	0.402	2	4	3	4	2	9	2	2	19	25	14	58	156	6	54	33	26
3	1	68	170	69	25	0.36	2	4	3	4	3	10	1	2	21	27	17	65	147	5	52	28	24
15	0	78	150	51.9	24	0.35	3	5	5	5	4	14	2	2	19	25	19	63	114	5	58	22	20
16	0	77	149	63	24	0.35	1	6	5	5	4	14	2	2	33	33	33	99	92	1	54	30	22

pt_basic_data.csv(基本情報を手動でまとめたもの)
pt_id	pt_sex	pt_age	pt_height	pt_weight	pt_foot_len	grip_power	exp	certified
1	1	50	191	87	28.5	44.5	270	1
2	1	28	173	87	27.5	49.1	73	0
3	1	28	168	67	27.5	46.2	13	0
4	0	24	164	65	24.5	34.9	43	0
7	0	34	160	52	24	31.3	139	1
15	1	22	178	59	26.5	40.5	18	0
"""