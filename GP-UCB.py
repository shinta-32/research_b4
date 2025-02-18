import os
import random
import json
import numpy as np
import time
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings

# numpy に warnings 属性がないため追加
np.warnings = warnings


#-------------------------------実験毎に設定する部分----------------------------------

#何のデータセット使用するか決める color_phone, color_car, Mii_physique, avatar_face
answer_format = 'avatar_face'
#実験を何周回すか決める
NUM_OF_ITERATION = 7
#一人に対して何回問い合わせするか決める
#アバター顔の場合20回、それ以外10回
NUM_OF_ANSWER = 20
#何人毎にクラスタリングするか決める
CLUSTERING_INTERVAL = 50
#何人分のデータで実験を行うかを決める。最大はcolor_h_sは291人、それ以外は300人
if answer_format == 'color_phone':
  NUM_OF_PEOPLE = 300
elif answer_format == 'color_car':
  NUM_OF_PEOPLE = 300
elif answer_format == 'Mii_physique':
  NUM_OF_PEOPLE = 300
elif answer_format == 'avatar_face':
  NUM_OF_PEOPLE = 300

#ファイル名を変えること！！

#--------------------------------------------------------------------------------------

if answer_format == 'color_phone':
    folder_path = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/phone' 
elif answer_format == 'color_car':
    folder_path = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/car' 
elif answer_format == 'Mii_physique':
    folder_path = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/Mii' 
elif answer_format == 'avatar_face':
    folder_path = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/avatar' 

from function_org import (
    get_train_data,
    get_SHAPE,
    get_X_TEST,
    get_SHAPE_ONE_DIM,
    GP_UCB
)

#ガウス過程回帰で使うカーネルとモデルを設定する
kernel = Matern(length_scale=10, nu=2.5) 
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

#ガウス過程回帰で使用する解空間を設定する
X_TEST = get_X_TEST(answer_format)
#計算に使用する変数
SHAPE = get_SHAPE(answer_format)
SHAPE_ONE_DIM = get_SHAPE_ONE_DIM(SHAPE)


#最終的なMAPVを入れるリストを作成する
MAPV = []

#------------------------------------------実験のメインの部分-----------------------------------------------------------
 
#start_time = time.time()
for a in range(NUM_OF_ITERATION):
  print('実験回数', a)
  #ファイルリストを入手してシャッフルする
  file_list = [f for f in os.listdir(folder_path) if f.endswith('.json')]
  random.shuffle(file_list)
  
  #各イテレーションごとに再使用する変数を設定・初期化する
  APV = []
  #１イテレーションの処理に入る
  for i in range(NUM_OF_PEOPLE):
    
    #ファイルを開いてデータを入手する操作
    file_path = folder_path + '/' + file_list[i]
    print(i, f"Reading file: {file_path}") 
    with open(file_path, 'r') as file:
      data = json.load(file)
      
    #x_train_truth, y_train_truthにデータを代入し、0～1に標準化する
    x_train_truth, y_train_truth = get_train_data(data, answer_format)
    max_y = np.max(y_train_truth)
    min_y = np.min(y_train_truth)
    if max_y == min_y:
        y_train_truth = np.array([0.5 for _ in range(len(y_train_truth))])  # 全て同じ値の場合は0.5に統一する
    else:
        y_train_truth = (y_train_truth - min_y) / (max_y - min_y) # 0~1に修正
        
    #GPRを使ってGround Truthを作成する
    gp_model.fit(x_train_truth, y_train_truth)
    mu_truth, sigma_truth = gp_model.predict(X_TEST.copy(), return_std=True)
    
    #t回ベイズ最適化を繰り返して分布を再現する
    x_trains_1dim = []
    for t in range(NUM_OF_ANSWER):
      #１．獲得関数から得た解のインデックスを保存する,さらにx_trains_1dimの末尾に追加する
      if t == 0:
        max_idx = random.randrange(0, SHAPE_ONE_DIM[0])
      else:
        max_idx = GP_UCB(t, NUM_OF_ANSWER, mu, sigma)
      x_trains_1dim.append(max_idx)
      
      #２．インデックスを多次元に直しproposed_paramに保存、問い合わせ結果をvalueに保存。
      if answer_format == 'color_phone':
        proposed_param = np.unravel_index(max_idx, (100, 360))
        proposed_param = tuple(reversed(proposed_param))
      elif answer_format == 'color_car':
        proposed_param = np.unravel_index(max_idx, (100, 360))
        proposed_param = tuple(reversed(proposed_param))
      else:
        proposed_param = np.unravel_index(max_idx, SHAPE)
      value = mu_truth[max_idx]
      
      #３．proposed_param, valueをそれぞれx_train, y_trainの末尾に追加する。
      if t == 0:
        x_train = np.array([proposed_param])
        y_train = np.array([value])
      else:
        proposed_param = np.array(proposed_param)
        x_train = np.block([[x_train], [proposed_param]])
        #np.appendだと時間がかかるため、list型に変形してからappendを実行して、その後にnp.array型に戻す
        y_train = y_train.tolist()     # リスト型に変換
        y_train.append(value)          # value を y_train に追加
        y_train = np.asarray(y_train)  # y_train を np.array型に直す
        
      #４．GPRで分布を更新する
      gp_model.fit(x_train, y_train)
      mu, sigma = gp_model.predict(X_TEST.copy(), return_std=True) 

    # 51人目以降のAPVをAPVリストの末尾に保存する。
    if i >= CLUSTERING_INTERVAL:
      APV.append(y_train)

  #250(241)人分のAPVの平均をとったMPAV（リスト形式）をMPAVリストの末尾に追加する
  APV_mean = np.mean(APV, axis=0)
  APV_mean = APV_mean.tolist()
  MAPV.append(APV_mean)
  print("MAPV", MAPV)


#全イテレーションのMAPVの平均をとる。 
MAPV_mean = np.mean(MAPV, axis=0)
MAPV_mean_list = MAPV_mean.tolist()
print("MAPV_mean_list", MAPV_mean_list)

#end_time = time.time()

# 取得したMAPV_meanとMAPVをまとめて保存する
result = {
    "MAPV_mean": MAPV_mean_list,
    "MAPV": MAPV
}

#取得したMAPV_meanをmapv.jsonに保存する
with open("log/GP_avatar_sep1.json", "a") as f:
  json.dump(result, f, indent=4)   

#print(end_time - start_time)