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
        
        delta_cols = [
            "gait_speed",
            "symmetry_index_sw",
            "stride_time",
            "stride_width",
            "hip_max_ext",
            "knee_max_flex",
            "ankle_max_do",
            "hip_max_ab",
        ]

        for col in delta_cols:
            pa_gait_params[f"{col}_delta"] = (
                pa_gait_params.loc[0, col]- min_assi_pa_gait_params.loc[0, col]
            )
    
        # # 介助による歩行速度の変化量を算出
        # speed_delta = pa_gait_params.loc[0, "gait_speed"] - min_assi_pa_gait_params.loc[0, "gait_speed"]
        
        pa_gait_header = pa_gait_params.columns.tolist()
        # # データシートに歩行速度変化量やpa_id, pt_idの列を追加
        # if "gait_speed_delta" not in pa_gait_header:
        #     pa_gait_params["gait_speed_delta"] = speed_delta
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
other_cols = [col for col in pa_gait_params_all.columns if col not in ['pa_id', 'pt_id']]
pa_gait_params_all = pa_gait_params_all[['pa_id', 'pt_id'] + other_cols]

# pa_id を若い順に並べ替え（pt_id も副次的にソート）
pa_gait_params_all = pa_gait_params_all.sort_values(by=['pa_id', 'pt_id']).reset_index(drop=True)

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
Combined dataframe shape: (10, 18)
   pa_id  pt_id  gait_speed  symmetry_index_sw  stride_time  stride_width  hip_max_ext  knee_max_flex  ...  gait_speed_delta  symmetry_index_sw_delta  stride_time_delta  stride_width_delta  hip_max_ext_delta  knee_max_flex_delta  ankle_max_do_delta  hip_max_ab_delta
0      2      2    0.820479          36.024845     1.394444      0.130753    14.657834      31.988449  ...          0.207650                 0.730727          -0.088889           -0.027308           7.213369             2.064087            4.122484          1.048003
1      2      3    0.801845          33.557047     1.275000      0.158555    14.025554      28.227694  ...          0.189016                -1.737071          -0.208333            0.000494           6.581089            -1.696668            7.983075          0.639962
2      3      2    0.677560          -4.651163     1.140000      0.084119    11.547005      50.993520  ...          0.027823               -20.867379          -0.090000            0.016890           3.349237             5.628085           -4.088897         -1.331511
3      3      4    0.774382          -1.474201     1.056667      0.089768     9.529688      55.913086  ...          0.124645               -17.690418          -0.173333            0.022540           1.331920            10.547651           -0.678167         -1.869345
4     15      1    0.647483          19.759926     1.450000      0.119784     6.292372      50.867716  ...          0.070722                -6.655168           0.056667           -0.022962           0.252016            -3.616657            0.804567         -1.287044
5     15      2    0.618756          14.953271     1.441667      0.115909     7.504814      61.572975  ...          0.041994               -11.461823           0.048333           -0.026838           1.464457             7.088602            1.703125          3.616176
6     15      4    0.600944          21.808511     1.370000      0.121461     8.618433      55.965452  ...          0.024182                -4.606584          -0.023333           -0.021285           2.578077             1.481079            2.789172          3.241003
7     16      1    0.642089           0.000000     1.196667      0.102134     4.836024      61.765990  ...          0.216303                 5.997392          -0.386667           -0.030205          -3.913165            -9.291973           -3.102713         -1.912153
8     16      7    0.521556           5.128205     1.288889      0.101805     7.062754      69.179726  ...          0.095770                11.125598          -0.294444           -0.030534          -1.686435            -1.878237            0.773471         -0.120668
9     16     15    0.624918           9.053498     1.127778      0.162339     3.175203      65.797471  ...          0.199132                15.050890          -0.455556            0.030000          -5.573987            -5.260492            8.266290         -0.311383

pt_assist_parameters.csv
Combined dataframe shape: (10, 10)
   pa_id  pt_id  hip_dist  wri_para_s  wri_nonpara_s   cos_sim  hip_cc_x  hip_cc_y  hip_cc_z  hip_cc_lag
0      2      2  0.415617    0.246079       0.438928  0.995653  0.968223  0.773644  0.891279    0.072222
1      2      3  0.483634    0.866011       0.356196  0.995228  0.965394  0.379778  0.915152   -0.041667
2      3      2  0.404926    0.244040       0.473516  0.998824  0.875149  0.322503  0.891676    0.020000
3      3      4  0.534947    0.956407       0.417257  0.998296  0.932241  0.486402  0.924868    0.036667
4     15      1  0.487691    0.847091       0.496528  0.999047  0.968007  0.809475  0.954468    0.008333
5     15      2  0.421240    0.475282       0.844142  0.998361  0.916869  0.654334  0.880372    0.087500
6     15      4  0.364007    0.396729       0.782628  0.996976  0.962212  0.682594  0.958852    0.020000
7     16      1  0.535572    0.906085       0.543996  0.996958  0.835561  0.686454  0.916692    0.046667
8     16      7  0.391030    0.762870       0.513139  0.998790  0.880973  0.613647  0.847930    0.061111
9     16     15  0.269103    0.370035       0.653955  0.986063  0.976583  0.751070  0.932824    0.036111

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