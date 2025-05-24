import json
import numpy as np
import pandas as pd

# ファイル名
file_name = "log3/CBOT_avatar_to_car_t1ratio_t2_x5_k001_minmaxthird.json"

# JSONファイルの読み込み
with open(file_name, "r") as f:
    data = json.load(f)

# All_cluster のデータを取得
all_cluster = data["All_cluster"]

# 各リストの変動係数（標準偏差 / 平均）を算出し、その平均をとる
iteration_cv_means = []
for iteration in all_cluster:
    cluster_cvs = []
    for cluster in iteration:
        for sublist in cluster:
            if len(sublist) > 1:
                mean = np.mean(sublist)
                std = np.std(sublist)
                if mean != 0:
                    cv = std / mean
                    cluster_cvs.append(cv)
    if cluster_cvs:
        iteration_cv_means.append(np.mean(cluster_cvs))

# 平均CVの計算
overall_mean_cv = np.mean(iteration_cv_means)

# DataFrameの作成（イテレーション別）
df = pd.DataFrame({
    "Iteration": list(range(1, len(iteration_cv_means) + 1)),
    "Mean CV": iteration_cv_means
})

# 平均行の追加
df.loc[len(df)] = ["Average", overall_mean_cv]

# 出力
print(file_name + "\n")
print(df.to_string(index=False))
