import random
import sys

# --- 設定項目 ---

# 扱う数値の全体の範囲
MIN_NUMBER = 0
MAX_NUMBER = 389

# --- スクリプト本体 ---

def generate_valid_lengths(target_sum, specs):
    """
    指定された合計値になるように、各範囲の仕様の範囲内で長さのリストを生成します。
    Args:
        target_sum (int): 目標とする合計の長さ。
        specs (list): 各範囲の仕様 [(min_len, max_len), ...]。
    Returns:
        list: 生成された長さのリスト。不可能な場合はNone。
    """
    min_possible_sum = sum(s[0] for s in specs)
    max_possible_sum = sum(s[1] for s in specs)
    if not (min_possible_sum <= target_sum <= max_possible_sum):
        print(f"エラー: 合計個数 {target_sum} は、指定された範囲仕様では生成不可能です。", file=sys.stderr)
        print(f"       この仕様で可能な合計個数の範囲は {min_possible_sum} ～ {max_possible_sum} です。", file=sys.stderr)
        return None

    lengths = []
    remaining_sum = target_sum
    
    # 後続の範囲が取りうる長さの合計をあらかじめ計算しておく
    min_suffix_sums = [0] * (len(specs) + 1)
    max_suffix_sums = [0] * (len(specs) + 1)
    for i in range(len(specs) - 1, -1, -1):
        min_suffix_sums[i] = specs[i][0] + min_suffix_sums[i+1]
        max_suffix_sums[i] = specs[i][1] + max_suffix_sums[i+1]

    # 各範囲の長さを決定していく
    for i, (min_len, max_len) in enumerate(specs):
        # 現在の範囲が取りうる長さの境界を計算する
        low_bound = max(min_len, remaining_sum - max_suffix_sums[i+1])
        high_bound = min(max_len, remaining_sum - min_suffix_sums[i+1])
        
        if low_bound > high_bound:
            # 基本的にこのエラーは発生しないはずだが、念のため
            return None 

        current_len = random.randint(low_bound, high_bound)
        lengths.append(current_len)
        remaining_sum -= current_len
        
    random.shuffle(lengths)
    return lengths

def find_non_overlapping_ranges(lengths, min_num, max_num):
    """
    指定された長さのリストに基づき、重複しない範囲を配置します。
    Args:
        lengths (list): 配置する各範囲の長さ。
        min_num (int): 配置可能な最小値。
        max_num (int): 配置可能な最大値。
    Returns:
        list: 生成された範囲のリスト [(start, end), ...]。失敗した場合はNone。
    """
    available_numbers = set(range(min_num, max_num + 1))
    ranges = []
    
    # 長い範囲から配置すると成功しやすい
    for length in sorted(lengths, reverse=True):
        # この長さの範囲を配置できる開始位置の候補を探す
        limit = max_num - length + 1
        possible_starts = [
            s for s in range(min_num, limit + 1)
            if all((s + i) in available_numbers for i in range(length))
        ]

        if not possible_starts:
            # 配置できる場所がない場合
            return None 

        start = random.choice(possible_starts)
        end = start + length - 1
        
        ranges.append((start, end))
        
        # 配置した数値を「使用済み」にする
        for i in range(start, end + 1):
            if i in available_numbers:
                available_numbers.remove(i)
                
    return sorted(ranges)

def main():
    """
    メイン処理
    """
    total_numbers = MAX_NUMBER - MIN_NUMBER + 1
    requested_length = int(total_numbers * 0.4)

    # --- 分析と解決策の説明 ---
    print("--- ご要望の分析 ---")
    print(f"全体の範囲: {MIN_NUMBER}～{MAX_NUMBER} (合計{total_numbers}個)")
    print(f"ご要望の合計個数: 全体の4割 = {requested_length}個")
    print("各範囲の個数: 20-30個が2つ以上, 10-20個が2つ以上, 1-10個が2つ以上")

    # --- 範囲仕様の動的生成 ---
    print("\n--- 範囲仕様の動的生成 ---")
    print(f"目標合計個数 {requested_length}個 を達成するため、各範囲の個数を調整します。")

    # 必須の範囲仕様
    base_specs = [
        (20, 30), (20, 30),
        (10, 20), (10, 20),
        (1, 10), (1, 10)
    ]
    # 追加可能な範囲の候補
    additional_spec_options = [(20, 30), (10, 20), (1, 10)]
    
    dynamic_specs = list(base_specs)

    # 範囲を追加して、目標合計個数を達成可能な最大値の範囲に収める
    max_possible_sum = sum(s[1] for s in dynamic_specs)
    while max_possible_sum < requested_length:
        spec_to_add = random.choice(additional_spec_options)
        dynamic_specs.append(spec_to_add)
        max_possible_sum += spec_to_add[1]

    # 最終的な最小合計が目標を超えていないかチェック
    min_possible_sum = sum(s[0] for s in dynamic_specs)
    if min_possible_sum > requested_length:
        print(f"\nエラー: 範囲の個数を調整しましたが、最小合計個数({min_possible_sum})が目標({requested_length})を超えてしまいました。", file=sys.stderr)
        print("       これは非常に稀なケースです。別の乱数シードで再試行してください。", file=sys.stderr)
        return

    # 生成された仕様のサマリーを表示
    spec_counts = {
        "20-30": dynamic_specs.count((20, 30)),
        "10-20": dynamic_specs.count((10, 20)),
        "1-10": dynamic_specs.count((1, 10)),
    }
    print("生成された範囲の仕様:")
    print(f"  - 20～30個の範囲: {spec_counts['20-30']}個")
    print(f"  - 10～20個の範囲: {spec_counts['10-20']}個")
    print(f"  -  1～10個の範囲: {spec_counts['1-10']}個")
    print(f"この仕様での合計個数の範囲: {min_possible_sum}個 ～ {max_possible_sum}個")
    print("-" * 25)

    # 合計個数を設定
    TARGET_TOTAL_LENGTH = requested_length
    
    # 1. 有効な長さの組み合わせを生成
    # 稀に失敗することがあるため、リトライ処理を入れる
    generated_lengths = None
    for _ in range(100): # 100回試行
        generated_lengths = generate_valid_lengths(TARGET_TOTAL_LENGTH, dynamic_specs)
        if generated_lengths is not None:
            break
    
    if generated_lengths is None:
        print("\nエラー: 有効な長さの組み合わせを生成できませんでした。")
        return
        
    # 2. 生成した長さを持つ範囲を重複なく配置
    # こちらも失敗する可能性があるのでリトライ
    generated_ranges = None
    for _ in range(100): # 100回試行
        generated_ranges = find_non_overlapping_ranges(generated_lengths, MIN_NUMBER, MAX_NUMBER)
        if generated_ranges is not None:
            break
    
    if generated_ranges is None:
        print("\nエラー: 範囲を重複なく配置できませんでした。")
        return

    # 3. 結果を表示
    print("\n--- 生成結果 ---")
    actual_total_length = 0
    for i, r in enumerate(generated_ranges):
        start, end = r
        length = end - start + 1
        actual_total_length += length
        print(f"範囲 {i+1}: {start:3d} ～ {end:3d} (長さ: {length:2d}個)")
    
    print("\n--- サマリー ---")
    print(f"生成された範囲のリスト: {generated_ranges}")
    print(f"各範囲の長さのリスト (ソート済): {sorted([r[1] - r[0] + 1 for r in generated_ranges], reverse=True)}")
    print(f"使用された範囲仕様 (個数): 20-30個: {spec_counts['20-30']}, 10-20個: {spec_counts['10-20']}, 1-10個: {spec_counts['1-10']}")
    print(f"合計個数: {actual_total_length} / {total_numbers}個")
    print(f"全体に占める割合: {actual_total_length / total_numbers:.2%}")

if __name__ == "__main__":
    main()



"""
--- ご要望の分析 ---
全体の範囲: 0～389 (合計390個)
ご要望の合計個数: 全体の4割 = 156個
各範囲の個数: 20-30個が2つ以上, 10-20個が2つ以上, 1-10個が2つ以上

--- 範囲仕様の動的生成 ---
目標合計個数 156個 を達成するため、各範囲の個数を調整します。
生成された範囲の仕様:
  - 20～30個の範囲: 2個
  - 10～20個の範囲: 3個
  -  1～10個の範囲: 4個
この仕様での合計個数の範囲: 74個 ～ 160個
-------------------------

--- 生成結果 ---
範囲 1:  17 ～  36 (長さ: 20個)
範囲 2:  46 ～  55 (長さ: 10個)
範囲 3:  67 ～  94 (長さ: 28個)
範囲 4: 180 ～ 197 (長さ: 18個)
範囲 5: 204 ～ 213 (長さ: 10個)
範囲 6: 240 ～ 269 (長さ: 30個)
範囲 7: 286 ～ 305 (長さ: 20個)
範囲 8: 308 ～ 317 (長さ: 10個)
範囲 9: 377 ～ 386 (長さ: 10個)

--- サマリー ---
生成された範囲のリスト: [(17, 36), (46, 55), (67, 94), (180, 197), (204, 213), (240, 269), (286, 305), (308, 317), (377, 386)]
各範囲の長さのリスト (ソート済): [30, 28, 20, 20, 18, 10, 10, 10, 10]
使用された範囲仕様 (個数): 20-30個: 2, 10-20個: 3, 1-10個: 4
合計個数: 156 / 390個
全体に占める割合: 40.00%
"""