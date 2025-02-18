import pandas as pd
import json
import re
import os

# Excelファイルを読み込む
excel_path = 'Batch_5280443_batch_results(reject).xlsx'
df = pd.read_excel(excel_path)

# 使用する列の範囲を定義
intensity_columns = [f'Answer.intensity_{i}' for i in range(78, 159)]
image_url_columns = [f'Input.image_url_{i}' for i in range(78, 159)]

# 相対パスで保存先のディレクトリを指定
output_directory = './exp_2412/avatar/'

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# URLから横幅(y)、縦幅(t)、目尻(z)、眉毛の傾き(k)を抽出する関数
def extract_avatar_features_from_url(url):
    # URLから "y90_t90_z50_k10" の部分を抽出
    features_match = re.search(r'y(\d+)_t(\d+)_z(\d+)_k(\d+)', url)
    if features_match:
        y_width = int(features_match.group(1))  # 横幅 (y)
        t_height = int(features_match.group(2))  # 縦幅 (t)
        z_edge = int(features_match.group(3))   # 目尻 (z)
        k_tilt = int(features_match.group(4))   # 眉毛の傾き (k)
        return y_width, t_height, z_edge, k_tilt
    return None, None, None, None

# 各WorkerIdに対して個別のJSONファイルを作成
for i in range(len(df)):
    worker_id = df['WorkerId'].iloc[i]  # WorkerIdを取得
    result = {}

    for intensity_col, image_col in zip(intensity_columns, image_url_columns):
        intensity_value = df[intensity_col].iloc[i]  # 嗜好度を取得
        image_url = df[image_col].iloc[i]  # 画像URLを取得

        # URLから横幅、縦幅、目尻、眉毛の傾きを抽出
        y_width, t_height, z_edge, k_tilt = extract_avatar_features_from_url(image_url)
        if y_width is not None and t_height is not None and z_edge is not None and k_tilt is not None:
            # 新しい画像ファイル名を特徴量で作成
            new_image_filename = f"images/avatar-{y_width}-{t_height}-{z_edge}-{k_tilt}.jpg"

            # 嗜好度をint型に変換して辞書に保存
            result[new_image_filename] = int(intensity_value)

    # 個別のJSONファイルを保存
    output_path = os.path.join(output_directory, f"{worker_id}_avatar.json")
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print(f"JSONファイルが{output_path}に保存されました")
