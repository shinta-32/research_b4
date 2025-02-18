import pandas as pd
import json
import re
import os

# Excelファイルを読み込む
excel_path = 'Batch_5280443_batch_results(reject).xlsx'
df = pd.read_excel(excel_path)

# 使用する列の範囲を定義
intensity_columns = [f'Answer.intensity_{i}' for i in range(29, 53)]
image_url_columns = [f'Input.image_url_{i}' for i in range(29, 53)]

# 相対パスで保存先のディレクトリを指定
output_directory = './exp_2412/car/'

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# URLからHSVを抽出する関数
def extract_hsv_from_url(url):
    # URLから "h270_s83_v100" の部分を抽出
    hsv_match = re.search(r'h(\d+)_s(\d+)_v(\d+)', url)
    if hsv_match:
        hue = int(hsv_match.group(1))      # 色相 (Hue)
        saturation = int(hsv_match.group(2))  # 彩度 (Saturation)
        value = int(hsv_match.group(3))      # 輝度 (Value)
        return hue, saturation, value
    return None, None, None

# 各WorkerIdに対して個別のJSONファイルを作成
for i in range(len(df)):
    worker_id = df['WorkerId'].iloc[i]  # WorkerIdを取得
    result = {}

    for intensity_col, image_col in zip(intensity_columns, image_url_columns):
        intensity_value = df[intensity_col].iloc[i]  # 嗜好度を取得
        image_url = df[image_col].iloc[i]  # 画像URLを取得

        # URLからHSV値を抽出
        hue, saturation, value = extract_hsv_from_url(image_url)
        if hue is not None and saturation is not None and value is not None:
            # 新しい画像ファイル名をHSV値で作成
            new_image_filename = f"images/car-{hue}-{saturation}-{value}.jpg"

            # 嗜好度をint型に変換して辞書に保存
            result[new_image_filename] = int(intensity_value)

    # 個別のJSONファイルを保存
    output_path = os.path.join(output_directory, f"{worker_id}_car.json")
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print(f"JSONファイルが{output_path}に保存されました")
