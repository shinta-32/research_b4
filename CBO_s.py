#CBOを自分の環境に合わせたもの
import os
import random
import json
import numpy as np
import pandas as pd
import time
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans
import warnings

# numpy に warnings 属性がないため追加
np.warnings = warnings

#-------------------------------実験毎に設定する部分----------------------------------

#何のデータセット使用するか決める color_phone, color_car, Mii_physique, avatar_face
answer_format = 'color_car'
#実験を何周回すか決める
NUM_OF_ITERATION = 7
#一人に対して何回問い合わせするか決める
#アバター顔の場合20回、それ以外10回
NUM_OF_ANSWER = 20
#何人毎にクラスタリングするか決める
CLUSTERING_INTERVAL = 50
#何人分のデータで実験を行うかを決める。
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
    GP_UCB,
    CBO,
    get_point_list_one_dim,
    get_point_list
)
#ガウス過程回帰で使うカーネルとモデルを設定する
kernel = Matern(length_scale=10, nu=2.5) 
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

#ガウス過程回帰で使用する解空間を設定する
X_TEST = get_X_TEST(answer_format)
#計算に使用する変数
SHAPE = get_SHAPE(answer_format)
SHAPE_ONE_DIM = get_SHAPE_ONE_DIM(SHAPE)

#分布をクラスタリングする時の代表点が入ったリスト
representative_point = get_point_list_one_dim(answer_format)

#最終的なMAPVを入れるリストを作成する
MAPV = []

#centersの解空間を表した点のリスト
point_list = get_point_list(answer_format)
point_list = np.array(point_list)

#------------------------------------------実験のメインの部分-----------------------------------------------------------
 
#start_time = time.time()
for a in range(NUM_OF_ITERATION):
  print('実験回数', a)
  # ファイルリストを入手してシャッフルする (jsonファイルのみを対象)
  file_list = [f for f in os.listdir(folder_path) if f.endswith('.json')]
  random.shuffle(file_list)
  
  #各イテレーションごとに再使用する変数を設定・初期化する
  APV = []
  APV_max = []
  mu_points = []
  mu_s = []
  #１イテレーションの処理に入る
  for i in range(NUM_OF_PEOPLE):
    
    #ファイルを開いてデータを入手する操作
    file_path = folder_path + '/' + file_list[i]

    print(i, f"Reading file: {file_path}") 
    with open(file_path, 'r', encoding='windows-1252') as file:
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
      if i < CLUSTERING_INTERVAL:
        if t == 0:
          max_idx = random.randrange(0, SHAPE_ONE_DIM[0])
        else:
          max_idx = GP_UCB(t, NUM_OF_ANSWER, mu, sigma)
      else:
        if t == 0:
          max_idx = random.randrange(0, SHAPE_ONE_DIM[0])
        else:
          max_idx = CBO(answer_format, t, NUM_OF_ANSWER, mu, sigma, x_trains_1dim, y_train, c_mu)
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
    
    #クラスタリングの処理を使う人のAPVをAPVリストの末尾に保存する、APVの最大値も保存する （51人目以降）
    if i >= CLUSTERING_INTERVAL:
      APV.append(y_train)
      APV_max.append(np.max(y_train))
    
    #クラスタリングのために、事後分布の代表点の期待値をmu_pointsに保存していく。 事後分布そのものもmu_sに保存しておく
    mu_points_individual = []
    for k in range(len(representative_point)):
      mu_points_individual.append(mu[representative_point[k]])
    mu_points.append(mu_points_individual)
    mu_s.append(mu)
    
    #x-meansでクラスタ数を決定する処理
    if (i+1) % CLUSTERING_INTERVAL == 0 and i != 0:
      initial_centers = kmeans_plusplus_initializer(mu_points, 2).initialize()
      xmeans_instance = xmeans(mu_points, initial_centers, kmax=10, ccore=False)  # kmaxは探索する最大クラスタ数
      xmeans_instance.process()
      clusters = xmeans_instance.get_clusters()
      #クラスタ数をnum_of_clusterに保存
      num_of_cluster = len(clusters)
      
    #k-means法でクラスタリングを行った後、クラスタ毎にクラスタの期待値を作成する
    if (i+1) >= CLUSTERING_INTERVAL and (i+1) % 10 == 0: 

      kmeans = KMeans(n_clusters= num_of_cluster)
      kmeans.fit(mu_points)
      cluster_number = kmeans.predict(mu_points)
      centers = kmeans.cluster_centers_
      
      #各クラスタのセントロイドの代表点centersにGPRを行い、各クラスタの期待値を復元する
      c_mu = [[] for i in range(len(centers))]
      #クラスタの期待値（c_mu[0], c_mu[1], …）を作成する
      for l in range(len(centers)):
        gp_model.fit(point_list, centers[l])
        c_mu[l], _ = gp_model.predict(X_TEST.copy(), return_std=True)
      c_mu = np.array(c_mu)
      
  #250(241)人分のAPV_maxを全イテレーション分保存する
  #with open("log/journal_experiment3/face_2.5_CBO_APVmax2.json", "a") as f:
  #  json.dump(APV_max, f)  
  
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

#取得したMAPV_meanをtest_mapv.jsonに保存する
with open("log/CBO_car_sep1_t20.json", "a") as f:
  json.dump(result, f, indent=4)   


#取得したMAPVをtest_mapv.jsonに保存する
#MAPV_list = MAPV.tolist() 
#with open("log/MAPV.json", "a") as f:
#  json.dump(MAPV, f)   
  
#print(end_time - start_time)
