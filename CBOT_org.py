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

#何のデータセット使用するか決める
# color_phone, color_car, Mii_physique, avatar_face　の4つ
answer_format_s = "avatar_face" #ソースドメイン
answer_format_t = "color_car"  #ターゲットドメイン
#実験を何周回すか決める
NUM_OF_ITERATION = 7
#一人に対して何回問い合わせするか決める
#アバター顔の場合20回、それ以外10回
NUM_OF_ANSWER = 10
#何人毎にクラスタリングするか決める
CLUSTERING_INTERVAL = 50
#何人分のデータで実験を行うかを決める
NUM_OF_PEOPLE = 300

#２回目以降の探索確認すること！
#ファイル名変えるのを忘れないこと！！（1番下）

#--------------------------------------------------------------------------------------

folder_path_s = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/avatar'
folder_path_t = os.path.dirname(os.path.abspath(__file__)) + '/exp_2412(result)/car'  

from function_org import (
    get_train_data,
    get_SHAPE,
    get_X_TEST,
    get_SHAPE_ONE_DIM,
    GP_UCB,
    CBO,
    CBOT_1,
    CBOT_2,
    get_point_list_one_dim,
    get_point_list
)
#ガウス過程回帰で使うカーネルとモデルを設定する
kernel = Matern(length_scale=10, nu=2.5) 
gp_model_s = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_model_t = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

#ガウス過程回帰で使用する解空間を設定する
X_TEST_s = get_X_TEST(answer_format_s)
X_TEST_t = get_X_TEST(answer_format_t)

#計算に使用する変数
SHAPE_s = get_SHAPE(answer_format_s)
SHAPE_t = get_SHAPE(answer_format_t)
SHAPE_ONE_DIM_s = get_SHAPE_ONE_DIM(SHAPE_s)
SHAPE_ONE_DIM_t = get_SHAPE_ONE_DIM(SHAPE_t)

#分布をクラスタリングする時の代表点が入ったリスト
representative_point_s = get_point_list_one_dim(answer_format_s)
representative_point_t = get_point_list_one_dim(answer_format_t)

#最終的なMAPVを入れるリストを作成する
MAPV = []

#全部のクラスタリングの様子を入れる
All_Cluster = []

#centersの解空間を表した点のリスト
point_list_s = get_point_list(answer_format_s)
point_list_s = np.array(point_list_s)

point_list_t = get_point_list(answer_format_t)
point_list_t = np.array(point_list_t)

#------------------------------------------実験のメインの部分-----------------------------------------------------------

#start_time = time.time()
for a in range(NUM_OF_ITERATION):
    print('実験回数', a)
    #ファイルリスト（ソースドメイン）を入手してシャッフルする(jsonファイルのみを対象)
    file_list_s =[f for f in os.listdir(folder_path_s) if f.endswith('.json')]
    random.shuffle(file_list_s)

    #ソースドメインを元にファイルリスト（ターゲットドメイン）を入手する(jsonファイルのみを対象)
    file_list_t = [f.replace('_avatar.json', '_car.json') for f in file_list_s if os.path.exists(os.path.join(folder_path_t, f.replace('_avatar.json', '_car.json')))]

    #各イテレーションごとに再使用する変数を設定・初期化する
    APV = []  #一人一人の問い合わせごとの嗜好値を格納（ターゲット）
    mu_points_s = []  #一人の代表点ごとの期待値を格納　クラスタリングに活用（ソース）
    mu_r_s = []  #GPRで予測されたすべての点での期待値を格納（ソース）
    mu_points_t = []  #一人の代表点ごとの期待値を格納　クラスタリングに活用（ターゲット）
    mu_r_t = []  #GPRで予測されたすべての点での期待値を格納(ターゲット)
    cluster_membership_rate = []  #クラスタの所属人数を保存する

    Cluster = [] #クラスタリングの様子を入れる

    #１イテレーションの処理に入る
    for i in range(NUM_OF_PEOPLE):

        #ファイル（ソースドメイン）を開いてデータを入手する操作
        file_path_s = folder_path_s + '/' + file_list_s[i]
        #ファイル（ターゲットドメイン）を開いてデータを入手する操作
        file_path_t = folder_path_t + '/' + file_list_t[i]

        print(a, i, f"Reading file: {file_path_s}{file_path_t}") 

        with open(file_path_s, 'r', encoding='windows-1252') as file_s:
            data_s = json.load(file_s)
        with open(file_path_t, 'r', encoding='windows-1252') as file_t:
            data_t = json.load(file_t)
        
        #（ソースドメイン）x_train_truth, y_train_truthにデータを代入し、0～1に標準化する
        x_train_truth_s, y_train_truth_s = get_train_data(data_s, answer_format_s)
        max_y_s = np.max(y_train_truth_s)
        min_y_s = np.min(y_train_truth_s)
        if max_y_s == min_y_s:
            y_train_truth_s = np.array([0.5 for _ in range(len(y_train_truth_s))])  # 全て同じ値の場合は0.5に統一する
        else:
            y_train_truth_s = (y_train_truth_s - min_y_s) / (max_y_s - min_y_s) # 0~1に修正
        
        #（ターゲットドメイン）x_train_truth, y_train_truthにデータを代入し、0～1に標準化する
        x_train_truth_t, y_train_truth_t = get_train_data(data_t, answer_format_t)
        max_y_t = np.max(y_train_truth_t)
        min_y_t = np.min(y_train_truth_t)
        if max_y_t == min_y_t:
            y_train_truth_t = np.array([0.5 for _ in range(len(y_train_truth_t))])  # 全て同じ値の場合は0.5に統一する
        else:
            y_train_truth_t = (y_train_truth_t - min_y_t) / (max_y_t - min_y_t) # 0~1に修正

        #GPRを使ってGround Truthを作成する（ソースドメイン）
        gp_model_s.fit(x_train_truth_s, y_train_truth_s)
        mu_truth_s, sigma_truth_s = gp_model_s.predict(X_TEST_s.copy(), return_std=True)

        #GPRを使ってGround Truthを作成する（ターゲットドメイン）
        gp_model_t.fit(x_train_truth_t, y_train_truth_t)
        mu_truth_t, sigma_truth_t = gp_model_t.predict(X_TEST_t.copy(), return_std=True)

        #s回ベイズ最適化を繰り返して分布を再現する(ソースドメイン)
        x_trains_1dim_s = []
        for s in range(NUM_OF_ANSWER):
            #１．獲得関数から得た解のインデックスを保存する,さらにx_trains_1dimの末尾に追加する
            if i < CLUSTERING_INTERVAL:
                if s == 0:
                    max_idx_s = random.randrange(0, SHAPE_ONE_DIM_s[0])
                else:
                    max_idx_s = GP_UCB(s, NUM_OF_ANSWER, mu_s, sigma_s)
            else:
                if s == 0:
                    max_idx_s = random.randrange(0, SHAPE_ONE_DIM_s[0])
                else:
                    max_idx_s = CBO(answer_format_s, s, NUM_OF_ANSWER, mu_s, sigma_s, x_trains_1dim_s, y_train_s, c_mu_s)
            x_trains_1dim_s.append(max_idx_s)

            #２．インデックスを多次元に直しproposed_paramに保存、問い合わせ結果をvalueに保存。
            if answer_format_s == 'color_car':
                proposed_param_s = np.unravel_index(max_idx_s, (100, 360))
                proposed_param_s = tuple(reversed(proposed_param_s))
            elif answer_format_s == 'color_phone':
                proposed_param_s = np.unravel_index(max_idx_s, (100, 360))
                proposed_param_s = tuple(reversed(proposed_param_s))
            else:
                proposed_param_s = np.unravel_index(max_idx_s, SHAPE_s)
            value_s = mu_truth_s[max_idx_s]

            #３．proposed_param, valueをそれぞれx_train, y_trainの末尾に追加する。
            if s == 0:
                x_train_s = np.array([proposed_param_s])
                y_train_s = np.array([value_s])
            else:
                proposed_param_s = np.array(proposed_param_s)
                x_train_s = np.block([[x_train_s], [proposed_param_s]])
                #np.appendだと時間がかかるため、list型に変形してからappendを実行して、その後にnp.array型に戻す
                y_train_s = y_train_s.tolist()     # リスト型に変換
                y_train_s.append(value_s)          # value を y_train に追加
                y_train_s = np.asarray(y_train_s)  # y_train を np.array型に直す
                
            #４．GPRで分布を更新する
            gp_model_s.fit(x_train_s, y_train_s)
            mu_s, sigma_s = gp_model_s.predict(X_TEST_s.copy(), return_std=True)


        #ソースドメインクラスタリング
        #クラスタリングのために、事後分布の代表点の期待値をmu_pointsに保存していく。 事後分布そのものもmu_rに保存しておく
        mu_points_individual_s = []
        for k in range(len(representative_point_s)):
            mu_points_individual_s.append(mu_s[representative_point_s[k]])
        mu_points_s.append(mu_points_individual_s)
        mu_r_s.append(mu_s)

        #一人ずつソースドメインではどこのクラスタか判別する
        if i >= CLUSTERING_INTERVAL:
            source_cluster_id = kmeans_s.predict([mu_points_s[-1]])[0]

        #x-meansでクラスタ数を決定する処理
        if (i+1) % 10 == 0 and i != 0:
            initial_centers_s = kmeans_plusplus_initializer(mu_points_s, 2).initialize()
            xmeans_instance_s = xmeans(mu_points_s, initial_centers_s, kmax=5, ccore=False)  # kmaxは探索する最大クラスタ数
            xmeans_instance_s.process()
            clusters_s = xmeans_instance_s.get_clusters()
            #クラスタ数をnum_of_clusterに保存
            num_of_cluster_s = len(clusters_s)

        #k-means法でクラスタリングを行った後、クラスタ毎にクラスタの期待値を作成する
        if (i+1) >= CLUSTERING_INTERVAL and (i+1) % 10 == 0:
            kmeans_s = KMeans(n_clusters= num_of_cluster_s)
            kmeans_s.fit(mu_points_s)
            cluster_number_s = kmeans_s.predict(mu_points_s)
            centers_s = kmeans_s.cluster_centers_

            #各クラスタのセントロイドの代表点centersにGPRを行い、各クラスタの期待値を復元する
            c_mu_s = [[] for i in range(len(centers_s))]
            #クラスタの期待値（c_mu[0], c_mu[1], …）を作成する
            for l in range(len(centers_s)):

                gp_model_s.fit(point_list_s, centers_s[l])
                c_mu_s[l], _ = gp_model_s.predict(X_TEST_s.copy(), return_std=True)

            c_mu_s = np.array(c_mu_s)



        #t回ベイズ最適化を繰り返して分布を再現する(ターゲットドメイン)
        x_trains_1dim_t = []
        for t in range(NUM_OF_ANSWER):
            #１．獲得関数から得た解のインデックスを保存する,さらにx_trains_1dimの末尾に追加する
            if i < CLUSTERING_INTERVAL:
                if t == 0:
                    max_idx_t = random.randrange(0, SHAPE_ONE_DIM_t[0])
                else:
                    max_idx_t = GP_UCB(t, NUM_OF_ANSWER, mu_t, sigma_t)
            else:
                if t == 0:
                    max_idx_t, membership_ratios = CBOT_1(answer_format_t, t, NUM_OF_ANSWER, mu_t, sigma_t, x_trains_1dim_t, y_train_t, c_mu_t, i, cluster_membership_rate, source_cluster_id)
                else:
                    max_idx_t = CBOT_2(answer_format_t, t, NUM_OF_ANSWER, mu_t, sigma_t, x_trains_1dim_t, y_train_t, c_mu_t, i, cluster_membership_rate, source_cluster_id, membership_ratios, mu_truth_t)
                    # max_idx_t = CBO(answer_format_t, t, NUM_OF_ANSWER, mu_t, sigma_t, x_trains_1dim_t, y_train_t, c_mu_t) 

            x_trains_1dim_t.append(max_idx_t)

            #２．インデックスを多次元に直しproposed_paramに保存、問い合わせ結果をvalueに保存。
            if answer_format_t == 'color_phone':
                proposed_param_t = np.unravel_index(max_idx_t, (100, 360))
                proposed_param_t = tuple(reversed(proposed_param_t))
            elif answer_format_t == 'color_car':
                proposed_param_t = np.unravel_index(max_idx_t, (100, 360))
                proposed_param_t = tuple(reversed(proposed_param_t))
            else:
                proposed_param_t = np.unravel_index(max_idx_t, SHAPE_t)
            value_t = mu_truth_t[max_idx_t]

            #３．proposed_param, valueをそれぞれx_train, y_trainの末尾に追加する。
            if t == 0:
                x_train_t = np.array([proposed_param_t])
                y_train_t = np.array([value_t])
            else:
                proposed_param_t = np.array(proposed_param_t)
                x_train_t = np.block([[x_train_t], [proposed_param_t]])
                #np.appendだと時間がかかるため、list型に変形してからappendを実行して、その後にnp.array型に戻す
                y_train_t = y_train_t.tolist()     # リスト型に変換
                y_train_t.append(value_t)          # value を y_train に追加
                y_train_t = np.asarray(y_train_t)  # y_train を np.array型に直す
                
            #４．GPRで分布を更新する
            gp_model_t.fit(x_train_t, y_train_t)
            mu_t, sigma_t = gp_model_t.predict(X_TEST_t.copy(), return_std=True)


        #クラスタリングの処理を使う人のAPVをAPVリストの末尾に保存する、APVの最大値も保存する （人目以降）
        if i >= CLUSTERING_INTERVAL:
            APV.append(y_train_t)

        #ターゲットドメインクラスタリング 
        #クラスタリングのために、事後分布の代表点の期待値をmu_pointsに保存していく。 事後分布そのものもmu_rに保存しておく
        mu_points_individual_t = []
        for k in range(len(representative_point_t)):
            mu_points_individual_t.append(mu_t[representative_point_t[k]])
        mu_points_t.append(mu_points_individual_t)
        mu_r_t.append(mu_t)

        #x-meansでクラスタ数を決定する処理
        if (i+1) % 10 == 0 and i != 0:
            initial_centers_t = kmeans_plusplus_initializer(mu_points_t, 2).initialize()
            xmeans_instance_t = xmeans(mu_points_t, initial_centers_t, kmax=5, ccore=False)  # kmaxは探索する最大クラスタ数
            xmeans_instance_t.process()
            clusters_t = xmeans_instance_t.get_clusters()
            #クラスタ数をnum_of_clusterに保存
            num_of_cluster_t = len(clusters_t)

        #k-means法でクラスタリングを行った後、クラスタ毎にクラスタの期待値を作成する
        if (i+1) >= CLUSTERING_INTERVAL and (i+1) % 10 == 0:
            kmeans_t = KMeans(n_clusters= num_of_cluster_t)
            kmeans_t.fit(mu_points_t)
            cluster_number_t = kmeans_t.predict(mu_points_t)
            centers_t = kmeans_t.cluster_centers_

            #各クラスタのセントロイドの代表点centersにGPRを行い、各クラスタの期待値を復元する
            c_mu_t = [[] for i in range(len(centers_t))]
            #クラスタの期待値（c_mu[0], c_mu[1], …）を作成する
            for l in range(len(centers_t)):

                gp_model_t.fit(point_list_t, centers_t[l])
                c_mu_t[l], _ = gp_model_t.predict(X_TEST_t.copy(), return_std=True)

            c_mu_t = np.array(c_mu_t)

        if (i+1) >= CLUSTERING_INTERVAL and (i+1) % 10 == 0:
            cluster_membership_rate = [[0 for _ in range(num_of_cluster_t)] for _ in range(num_of_cluster_s)]
            for s in range(len(mu_points_s)):
              for t in range(len(mu_points_t)):
                  # kmeans_s = KMeans(n_clusters=len(centers_s))
                  # kmeans_s.fit(mu_points_s)  # 既存の mu_points_t を使ってフィット
                  source_cluster_id = kmeans_s.predict([mu_points_s[s]])[0]

                  # kmeans_t = KMeans(n_clusters=len(centers_t))
                  # kmeans_t.fit(mu_points_t)  # 既存の mu_points_t を使ってフィット
                  target_cluster_id = kmeans_t.predict([mu_points_t[t]])[0]

                  if s == t:
                      cluster_membership_rate[source_cluster_id][target_cluster_id] += 1
            Cluster.append(cluster_membership_rate)

    #250(241)人分のAPVの平均をとったMPAV（リスト形式）をMPAVリストの末尾に追加する
    APV_mean = np.mean(APV, axis=0)
    APV_mean = APV_mean.tolist()
    MAPV.append(APV_mean)
    print("MAPV", MAPV)

    All_Cluster.append(Cluster)
    print("All_Cluster", All_Cluster)

#全イテレーションのMAPVの平均をとる。 
MAPV_mean = np.mean(MAPV, axis=0)
MAPV_mean_list = MAPV_mean.tolist()
print("MAPV_mean_list", MAPV_mean_list)

#end_time = time.time()

# 取得したMAPV_meanとMAPVをまとめて保存する
result = {
    "MAPV_mean": MAPV_mean_list,
    "MAPV": MAPV,
    "All_cluster": All_Cluster
}

#取得したMAPV_meanをtest_mapv.jsonに保存する
with open("log3/CBOT_avatar_to_car_t1ratio_t2_x5_k001_minmaxfourth.json", "w") as f:
  json.dump(result, f, indent=4)   







