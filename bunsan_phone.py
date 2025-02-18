import os
import json
import numpy as np

# 指定されたディレクトリ内の全てのJSONファイルを読み込み、
# 各ファイルのデータを正規化して分散を計算し、それらの平均を求める関数
def calculate_variance_mean(directory):
    # 各ファイルから計算した分散を格納するリストを初期化
    all_variances = []

    # ディレクトリ内の全てのファイルを繰り返し処理
    for filename in os.listdir(directory):
        # 拡張子が".json"のファイルのみ処理
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            # JSONデータを読み込む
            with open(filepath, 'r') as f:
                data = json.load(f)

            # 値を抽出
            values = np.array(list(data.values()))
            if len(values) > 1:
                max_value = values.max()
                min_value = values.min()
                if max_value != min_value:
                    # 最大値と最小値が異なる場合のみ正規化を実施
                    normalized_values = (values - min_value) / (max_value - min_value)
                else:
                    # 最大値と最小値が同じ場合は全ての値が一定のため、分散は0
                    normalized_values = np.zeros_like(values)
            else:
                # 値が1つだけの場合はそのまま使用
                normalized_values = values

            # 正規化された値で分散を計算
            variance = np.var(normalized_values)

            # 計算した分散をリストに追加
            all_variances.append(variance)

    # 分散のリストが空でない場合は平均を計算
    if all_variances:
        variance_mean = np.mean(all_variances)
    else:
        variance_mean = None

    return variance_mean

# 使用例
directory_path = "exp_2412(result)/avatar/"  # 実際のディレクトリパスに置き換えてください
mean_variance = calculate_variance_mean(directory_path)

# 結果を出力
if mean_variance is not None:
    print(f"全てのJSONファイルの正規化後の分散の平均値は: {mean_variance}")
else:
    print("ディレクトリ内に有効なJSONファイルが見つかりませんでした。")
