"""
Motive, IMU, GoProすべての衝突フレームの検出時間を比較する
"""

from pathlib import Path
import json

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    root_dir = Path(r"G:\gait_pattern\20250915_synctest")
    check_id_list = ["3","4","6"]
    for id in check_id_list:
        print(f"\n=== ID: {id} の衝突フレーム情報 ===")
        
        gopro_json_path = root_dir / f"{id}_gopro_impact_info.json"
        gopro_json = load_json(gopro_json_path)
        
        imu_json_path = root_dir / f"{id}_imu_impact_info.json"
        imu_json = load_json(imu_json_path)
        
        motive_json_path = root_dir / f"10{id}_motive_impact_info.json"
        motive_json = load_json(motive_json_path)
        
        print(f"gopro_impact_info: {gopro_json}, "
              f"imu_impact_info: {imu_json}, "
              f"motive_impact_info: {motive_json}")
        
        # imuの発光から衝突までのフレームを基準としてみる
        diff_time_gopro_imu = gopro_json.get("impact_time_ledbase")*1000 - imu_json.get("elapsed_time_ms")
        diff_time_motive_imu = ((motive_json.get("impact_frame_number") + imu_json.get("diff_port0to1_frame")) - imu_json.get("impact_frame_number")) * 10
        print(f"-> imu基準でのGoProとの差: {diff_time_gopro_imu}ms, Motiveとの差: {diff_time_motive_imu}ms")
        
        diff_time_gopro_motive = gopro_json.get("impact_time_ledbase")*1000 - ((motive_json.get("impact_frame_number") + imu_json.get("diff_port0to1_frame")) * 10)
        diff_time_imu_motive = -diff_time_motive_imu    
        print(f"-> motive基準でのGoProとの差: {diff_time_gopro_motive}ms, IMUとの差: {diff_time_imu_motive}ms")
        
        diff_time_imu_gopro = -diff_time_gopro_imu
        diff_time_motive_gopro = -diff_time_gopro_motive
        print(f"-> gopro基準でのIMUとの差: {diff_time_imu_gopro}ms, Motiveとの差: {diff_time_motive_gopro}ms")



if __name__ == "__main__":
    main()