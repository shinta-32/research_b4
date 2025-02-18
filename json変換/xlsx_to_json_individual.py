import pandas as pd
import json
import os

# Excelファイルを読み込む
file_path = 'Batch_5280443_batch_results(reject).xlsx'
df = pd.read_excel(file_path)

# 保存先ディレクトリの指定
output_directory = './exp_2412/individual/'

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 必要な列を定義
columns = ['Answer.age', 'Answer.gender', 'Answer.region', 'Answer.mbti', 'WorkerId']

# 各WorkerIdごとにJSONファイルを作成
for i in range(len(df)):
    worker_id = df['WorkerId'].iloc[i]
    age = df['Answer.age'].iloc[i]
    gender = df['Answer.gender'].iloc[i]
    region = df['Answer.region'].iloc[i]
    mbti = df['Answer.mbti'].iloc[i]

    # 辞書形式でデータをまとめる
    individual_data = {
        "age": age,
        "gender": gender,
        "region": region,
        "mbti": mbti
    }

    # JSONファイルの保存
    output_path = os.path.join(output_directory, f"{worker_id}_individual.json")
    with open(output_path, 'w') as json_file:
        json.dump(individual_data, json_file, indent=4)

    print(f"JSONファイルが{output_path}に保存されました")
