#必要な関数の定義（GPR,BO, クラスタリングに使う）

import numpy as np
import re
import math
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

#生データをnp.array形式のx_trainとy_trainに変換する関数  
def get_train_data(data, answer_format):
    for i, keyValue in enumerate(data.items()):
        key, value = keyValue
        if answer_format == "color_phone":
            m = re.match(r"images/phone-(\d+)-(\d+)-100.jpg", key)
            _h, _s = m.groups()
            h, s = int(_h), int(_s)
            new_hs_pair = np.array([h, s])
            if i == 0:
                x_train = np.array([new_hs_pair])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_hs_pair]])
                y_train = np.append(y_train, value)
        elif answer_format == "color_car":
            m = re.match(r"images/car-(\d+)-83-(\d+).jpg", key)
            _h, _s = m.groups()
            h, s = int(_h), int(_s)
            new_hs_pair = np.array([h, s])
            if i == 0:
                x_train = np.array([new_hs_pair])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_hs_pair]])
                y_train = np.append(y_train, value)
        elif answer_format == "Mii_physique":
            m = re.match(r"images/Mii-(\d+)-(\d+).jpg", key)
            _h, _w = m.groups()
            h, w = int(_h), int(_w)
            new_hw_pair = np.array([h, w])
            if i == 0:
                x_train = np.array([new_hw_pair])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_hw_pair]])
                y_train = np.append(y_train, value)
        elif answer_format == "avatar_face":
            m = re.match(r"images/avatar-(\d+)-(\d+)-(\d+)-(\d+).jpg", key)
            _y, _t, _z, _k = map(int, m.groups())
            # action unitの値0～1を0～20に変換して21個の整数として扱う
            y, t, z, k = int(_y / 5), int(_t / 5), int(_z / 5), int(_k / 5)
            new_af_unit = np.array([y, t, z, k])
            if i == 0:
                x_train = np.array([new_af_unit])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_af_unit]])
                y_train = np.append(y_train, value)
        elif answer_format == "action_unit_4":
            png_name = key.split('/')[-1]
            m = re.match(r"au4-(\d+\.\d+)_au7-(\d+\.\d+)_au9-(\d+\.\d+)_au15-(\d+\.\d+).png", png_name)
            _au4, _au7, _au9, _au15 = map(float, m.groups())
            # action unitの値0～1を0～20に変換して21個の整数として扱う
            au4, au7, au9, au15 = int(20 * _au4), int(20 * _au7), int(20 * _au9), int(20 * _au15)
            new_au_unit = np.array([au4, au7, au9, au15])
            if i == 0:
                x_train = np.array([new_au_unit])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_au_unit]])
                y_train = np.append(y_train, value)
        elif answer_format == "voice_3":
            wav_name = key.split('/')[-1]
            m = re.match(r"(\d+)_(\-*\d+)_(\d+).wav", wav_name)
            speed, pitch, happiness = map(int, m.groups())
            # speed:50～200 -> 0～75
            speed = (speed - 50) // 2
            # pitch:-150～150 -> 0～150
            pitch = (pitch + 150) // 2
            # happiness:0～100 -> 0～50
            happiness = happiness // 2
            new_voice_unit = np.array([speed, pitch, happiness])
            if i == 0:
                x_train = np.array([new_voice_unit])
                y_train = np.array([value])
            else:
                x_train = np.block([[x_train], [new_voice_unit]])
                y_train = np.append(y_train, value)
    return x_train, y_train


#それぞれのデータセットに対応する解空間を表すタプルを作成する関数   get_X_TESTに使用する
def get_SHAPE(answer_format):
    if answer_format == "color_phone":
        return (360, 100)
    elif answer_format == "color_car":
        return (360, 100)
    elif answer_format == "Mii_physique":
        return (100, 100)
    elif answer_format == "avatar_face":
        return (21, 21, 21, 21)

#それぞれのデータセットの解空間と同じ広さを持つnp.array形式のリストを作成する関数　
def get_X_TEST(answer_format: str):
    SHAPE = get_SHAPE(answer_format)
    if answer_format == "color_phone":
        X_TEST_h = np.arange(SHAPE[0])  # Hue
        X_TEST_s = np.arange(SHAPE[1])  # Saturation
        X1, X2 = np.meshgrid(X_TEST_h, X_TEST_s)
        X1 = X1.reshape(SHAPE[0] * SHAPE[1], 1)
        X2 = X2.reshape(SHAPE[0] * SHAPE[1], 1)
        return np.concatenate([X1, X2], 1)
    elif answer_format == "color_car":
        X_TEST_h = np.arange(SHAPE[0])  # Hue
        X_TEST_s = np.arange(SHAPE[1])  # Saturation
        X1, X2 = np.meshgrid(X_TEST_h, X_TEST_s)
        X1 = X1.reshape(SHAPE[0] * SHAPE[1], 1)
        X2 = X2.reshape(SHAPE[0] * SHAPE[1], 1)
        return np.concatenate([X1, X2], 1)
    else:
        X_test = []
        # 再帰関数でX_testを作成する
        def make_X_test(current_data, SHAPE, current_shape_idx):
            if current_shape_idx == len(SHAPE):
                X_test.append(np.array(current_data))
                return
            for i in range(SHAPE[current_shape_idx]):
                next_data = current_data.copy()
                next_data.append(i)
                make_X_test(next_data, SHAPE, current_shape_idx + 1)

        make_X_test([], SHAPE, 0)
        return np.array(X_test)


#多次元のnp.ndarrayリストを1次元に変換する関数　SHAPE_ONE_DIMを作成するのに使う
def get_SHAPE_ONE_DIM(shape):
    SHAPE_ONE_DIM = [1,]
    for i in range(len(shape)):
        SHAPE_ONE_DIM[0] *= shape[i]
    return tuple(SHAPE_ONE_DIM)


#GP-UCBの獲得関数を定義し、その最大値を返す関数
def GP_UCB(num_answer, num_of_answer, mu, sigma):
  alpha = 10 * 10 ** (-4 * num_answer / num_of_answer)
  #alpha = 10 * 10 ** (-2.5 * num_answer / num_of_answer)
  #alpha = 1
  acquisition = mu + alpha * sigma
  return np.argmax(acquisition)


def CBO(answer_format, num_answer, num_of_answer, mu, sigma, x_trains_1dim, y_train, c_mu):
    alpha_1 = 10 * 10 ** (-4 * num_answer / num_of_answer)
    alpha_2 =  5 * 10 ** (-1.15 * num_answer / num_of_answer)
    #alpha_1 = 10 * 10 ** (-2.5 * num_answer / num_of_answer)
    #alpha_2 =  5 * 10 ** (-0.85 * num_answer / num_of_answer)

    #得られた評価値と各c_muの距離の合計を測って記録する()
    errors = np.zeros(len(c_mu))
    for j in range(len(c_mu)):
      for h in range(len(y_train)):
        errors[j] += (y_train[h] - c_mu[j][x_trains_1dim[h]]) ** 2

    #距離の合計が小さいほうを所属クラスタと認定し、そのインデックスを取得する
    belong_cluster_number = np.argmin(errors)
    acquisition = mu + alpha_1 * sigma + alpha_2 * c_mu[belong_cluster_number]
    return np.argmax(acquisition)


#１回目の問い合わせ(所属率を計算して問い合わせ)
def CBOT_1(answer_format, num_answer, num_of_answer, mu, sigma, x_trains_1dim, y_train, c_mu_t, i, cluster_membership_rate, source_cluster_id):
    alpha_1 = 1
    alpha_2 = 1

    # alpha_1 = 10 * 10 ** (-4 * num_answer / num_of_answer)
    # alpha_2 =  5 * 10 ** (-1.15 * num_answer / num_of_answer)
    #alpha_1 = 10 * 10 ** (-2.5 * num_answer / num_of_answer)
    #alpha_2 =  5 * 10 ** (-0.85 * num_answer / num_of_answer)


    # 確率的
    membership_ratios = np.zeros(len(cluster_membership_rate[source_cluster_id]), dtype=float)  #所属率を保存
    c_mu_total = np.zeros_like(c_mu_t[0])  # 全体の期待値を初期化
    for t_idx in range(len(cluster_membership_rate[source_cluster_id])):
        membership_ratios[t_idx] = cluster_membership_rate[source_cluster_id][t_idx] / sum(cluster_membership_rate[source_cluster_id])
        # c_mu_total += membership_ratios[t_idx] * c_mu_t[t_idx]  # 要素ごとの積を計算して合計

        # --- 正規化処理（min-maxスケーリング） ---
        minmax_c_mu = c_mu_t[t_idx]
        if np.max(minmax_c_mu) == np.min(minmax_c_mu):
            normalized_c_mu = np.ones_like(minmax_c_mu) * 0.5  # すべて同じ値の場合は0.5で統一
        else:
            normalized_c_mu = (minmax_c_mu - np.min(minmax_c_mu)) / (np.max(minmax_c_mu) - np.min(minmax_c_mu))
        c_mu_total += membership_ratios[t_idx] * normalized_c_mu
        
    print('source_cluster_id', source_cluster_id)
    print('cluster_membership_rate', cluster_membership_rate)
    print('membership_ratios', membership_ratios)
    print("c_mu_total", c_mu_total)


    # ハード選択
    # print('source_cluster_id', source_cluster_id)
    # print('cluster_membership_rate', cluster_membership_rate)
    # membership_ratios = np.zeros(len(cluster_membership_rate[source_cluster_id]), dtype=float)  #所属率を保存
    # c_mu_total = np.zeros_like(c_mu_t[0])  # 全体の期待値を初期化
    # for t_idx in range(len(cluster_membership_rate[source_cluster_id])):
    #     membership_ratios[t_idx] = cluster_membership_rate[source_cluster_id][t_idx] / sum(cluster_membership_rate[source_cluster_id])
    # max_target_idx = np.argmax(cluster_membership_rate[source_cluster_id])
    # print('max_target_idx', max_target_idx)

    #全体の式
    #確率的
    # acquisition = mu + alpha_1 * sigma + alpha_2 * c_mu_total
    acquisition = c_mu_total

    #ハード選択
    # acquisition = mu + alpha_1 * sigma + alpha_2 * c_mu_t[max_target_idx]
    # acquisition = c_mu_t[max_target_idx]

    return np.argmax(acquisition), membership_ratios


#２回目以降の問い合わせ
def CBOT_2(answer_format, num_answer, num_of_answer, mu, sigma, x_trains_1dim, y_train, c_mu_t, i, cluster_membership_rate, source_cluster_id, membership_ratios,  mu_truth_t):
    # alpha_1 = 1
    # alpha_2 = 1
    k = 0.01

    alpha_1 = 10 * 10 ** (-4 * num_answer / num_of_answer)
    alpha_2 =  5 * 10 ** (-1.15 * num_answer / num_of_answer)
    #alpha_1 = 10 * 10 ** (-2.5 * num_answer / num_of_answer)
    #alpha_2 =  5 * 10 ** (-0.85 * num_answer / num_of_answer)

    inquiry_distance = np.zeros(len(membership_ratios), dtype=float)   #1つ前の問い合わせ時のGround Truthとの差を保存（クラスターの数だけ）
    c_mu_total = np.zeros_like(c_mu_t[0])  # 全体の期待値を初期化
    for j in range(len(membership_ratios)):
      for h in range(len(y_train)):
        inquiry_distance[j] += abs(mu_truth_t[x_trains_1dim[h]] - c_mu_t[j][x_trains_1dim[h]]) #差を正にして、それぞれのクラスタごとに足し合わせる
        

    # (1/inquiry_distance[j] + 1) を計算
    membership_updates = np.zeros(len(membership_ratios), dtype=float)
    for j in range(len(membership_ratios)):
        membership_updates[j] = membership_ratios[j] + (1 / (inquiry_distance[j] + k))
    print('membership_updates', membership_updates)

    # 全体の正規化
    membership_sum = np.sum(membership_updates)  # 全体の総和を計算
    for j in range(len(membership_ratios)):
        membership_ratios[j] = membership_updates[j] / membership_sum  # 正規化して確率化
        # c_mu_total += membership_ratios[j] * c_mu_t[j]  # 要素ごとの積を計算して合計

        # --- 正規化処理（min-maxスケーリング） ---
        minmax_c_mu = c_mu_t[j]
        if np.max(minmax_c_mu) == np.min(minmax_c_mu):
            normalized_c_mu = np.ones_like(minmax_c_mu) * 0.5  # すべて同じ値の場合は0.5で統一
        else:
            normalized_c_mu = (minmax_c_mu - np.min(minmax_c_mu)) / (np.max(minmax_c_mu) - np.min(minmax_c_mu))
        c_mu_total += membership_ratios[j] * normalized_c_mu

    print('membership_ratios', membership_ratios)

    # 全体の式
    acquisition = mu + alpha_1 * sigma + alpha_2 * c_mu_total


    return np.argmax(acquisition)


#クラスタリングの際に使う代表点の一次元リストを取得する関数　
def get_point_list_one_dim(answer_format):
  point_list = []
  point_1dim = []
  point_list_1dim = []
  if answer_format == 'color_phone':
    for s in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      # for h in [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329]:
      for h in [0, 22, 44, 67, 89, 112, 134, 157, 179, 202, 224, 247, 269, 292, 314, 337]:
        point_list.append([h, s])
    for i in range(len(point_list)):
      point_1dim = point_list[i][0] + 360 * point_list[i][1]
      point_list_1dim.append(point_1dim)
  elif answer_format == 'color_car':
    for s in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      # for h in [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329]:
      for h in [0, 22, 44, 67, 89, 112, 134, 157, 179, 202, 224, 247, 269, 292, 314, 337]:
        point_list.append([h, s])
    for i in range(len(point_list)):
      point_1dim = point_list[i][0] + 360 * point_list[i][1]
      point_list_1dim.append(point_1dim)
  elif answer_format == 'Mii_physique':
    for w in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      for h in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
        point_list.append([h, w])
    for i in range(len(point_list)):
      point_1dim = point_list[i][0] + 100 * point_list[i][1]
      point_list_1dim.append(point_1dim)
  elif answer_format == 'avatar_face':
    for a in [0, 5, 10, 15, 20]:
      for b in [0, 5, 10, 15, 20]:
        for c in [0, 5, 10, 15, 20]:
          for d in [0, 5, 10, 15, 20]:
            point_list.append([a, b, c, d])
    for i in range(len(point_list)):
      point_1dim = point_list[i][3] + 21 * point_list[i][2] + 441 * point_list[i][1] + 9261 * point_list[i][0]
      point_list_1dim.append(point_1dim)
  elif answer_format == 'action_unit_4':
    for a in [1, 7, 13, 19]:
      for b in [3, 10, 17]:
        for c in [1, 7, 13, 19]:
          for d in [3, 10, 17]:
            point_list.append([a, b, c, d])
    for i in range(len(point_list)):
      point_1dim = point_list[i][3] + 21 * point_list[i][2] + 441 * point_list[i][1] + 9261 * point_list[i][0]
      point_list_1dim.append(point_1dim)
  elif answer_format == 'voice_3':
    for speed in [8, 23, 38, 53, 68]:
      for pitch in [0, 30, 60, 90, 120, 150]:
        for happiness in [7, 19, 31, 43]:
          point_list.append([speed, pitch, happiness])
    for i in range(len(point_list)):
      point_1dim = point_list[i][2] + 51 * point_list[i][1] + 7701 * point_list[i][0]
      point_list_1dim.append(point_1dim)
  return  point_list_1dim


#クラスタ中心をGPRで再現する際に使う関数
def get_point_list(answer_format):
  point_list = []
  if answer_format == 'color_phone':
    for s in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      # for h in [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329]:
      for h in [0, 22, 44, 67, 89, 112, 134, 157, 179, 202, 224, 247, 269, 292, 314, 337]:
        point_list.append([h, s])
  elif answer_format == 'color_car':
    for s in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      # for h in [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329]:
      for h in [0, 22, 44, 67, 89, 112, 134, 157, 179, 202, 224, 247, 269, 292, 314, 337]:
        point_list.append([h, s])
  elif answer_format == 'Mii_physique':
    for w in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
      for h in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
        point_list.append([h, w])
  elif answer_format == 'avatar_face':
    for a in [0, 5, 10, 15, 20]:
      for b in [0, 5, 10, 15, 20]:
        for c in [0, 5, 10, 15, 20]:
          for d in [0, 5, 10, 15, 20]:
            point_list.append([a, b, c, d])
  elif answer_format == 'action_unit_4':
    for a in [1, 7, 13, 19]:
      for b in [3, 10, 17]:
        for c in [1, 7, 13, 19]:
          for d in [3, 10, 17]:
            point_list.append([a, b, c, d])
  elif answer_format == 'voice_3':
    for speed in [8, 23, 38, 53, 68]:
      for pitch in [0, 30, 60, 90, 120, 150]:
        for happiness in [7, 19, 31, 43]:
          point_list.append([speed, pitch, happiness])
  return point_list

