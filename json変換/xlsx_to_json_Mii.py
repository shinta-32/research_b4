import pandas as pd
import json
import re
import os

# Excelファイルを読み込む
excel_path = 'Batch_5280443_batch_results(reject).xlsx'
df = pd.read_excel(excel_path)

# 使用する列の範囲を定義
intensity_columns = [f'Answer.intensity_{i}' for i in range(53, 78)]
image_url_columns = [f'Input.image_url_{i}' for i in range(53, 78)]

# 相対パスで保存先のディレクトリを指定
output_directory = './exp_2412/Mii/'

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# URLから背丈(h)と幅(w)を抽出する関数
def extract_hw_from_url(url):
    # URLから "h75_w75" の部分を抽出
    hw_match = re.search(r'h(\d+)_w(\d+)', url)
    if hw_match:
        height = int(hw_match.group(1))  # 背丈 (Height)
        width = int(hw_match.group(2))  # 幅 (Width)
        return height, width
    return None, None

# 各WorkerIdに対して個別のJSONファイルを作成
for i in range(len(df)):
    worker_id = df['WorkerId'].iloc[i]  # WorkerIdを取得
    result = {}

    for intensity_col, image_col in zip(intensity_columns, image_url_columns):
        intensity_value = df[intensity_col].iloc[i]  # 嗜好度を取得
        image_url = df[image_col].iloc[i]  # 画像URLを取得

        # URLから背丈(h)と幅(w)を抽出
        height, width = extract_hw_from_url(image_url)
        if height is not None and width is not None:
            # 新しい画像ファイル名を背丈と幅で作成
            new_image_filename = f"images/Mii-{height}-{width}.jpg"

            # 嗜好度をint型に変換して辞書に保存
            result[new_image_filename] = int(intensity_value)

    # 個別のJSONファイルを保存
    output_path = os.path.join(output_directory, f"{worker_id}_Mii.json")
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print(f"JSONファイルが{output_path}に保存されました")
