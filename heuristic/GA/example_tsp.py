import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from GA import PermutationDomain

SEED = 42
np.random.seed(SEED)

current_dir = os.path.dirname(os.path.abspath(__file__))

class TSP:
    def __init__(self, dis_matrix):
        self.dis_matrix = dis_matrix

    def cal_distance(self, path):
        path = np.hstack([path, path[..., [0]]]) # buile ring
        return np.sum(self.dis_matrix[path[..., :-1], path[..., 1:]], axis=-1)


class CityTSP(TSP):
    def __init__(self, csv_path, size=None):
        data = pd.read_csv(csv_path, delimiter=';', header=None).values
        self.pos = data[:size, 1:].astype(float)
        self.dis_matrix = pairwise_distances(self.pos, metric='euclidean')
        self.city_list = np.char.strip(data[:size, 0].astype(str))
        self.n = len(self.city_list)


    def get_city_name(self, idx):
        assert 0 <= idx < len(self.city_list)
        return self.city_list[idx]

    def draw_path(self, path):
        fig, ax = plt.subplots()
        ax.set_title("TSP Path")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # 绘制城市点
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c='blue')

        # 绘制路径
        for i in range(len(path)):
            start_city = self.pos[path[i]]
            end_city = self.pos[path[(i + 1) % len(path)]]
            ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], c='red')

        # 添加城市名称
        for i, city in enumerate(path):
            ax.text(self.pos[city][0], self.pos[city][1], i) # , fontsize=8, ha='right')

        # plt.legend()
        plt.show()

class HardCityTSP(CityTSP):
    def __init__(self, csv_path, size=None):
        data = pd.read_csv(csv_path, delimiter=',', header=None).values
        max_size = data.shape[0]
        indices = np.random.choice(max_size, size=size, replace=False) if size else np.arange(max_size)
        
        self.pos = data[indices, -2:].astype(float)
        self.dis_matrix = pairwise_distances(self.pos, metric='euclidean')
        self.city_list = np.array(['-'.join(np.char.strip(row)) for row in data[indices, :3].astype(str)])
        self.n = len(self.city_list)


def GA(domain: PermutationDomain, fun_fitness, num=50, n_step=50):
    group = domain.gen(num)
    n_city = domain.n
    with tqdm(total=n_step, desc='GA') as pbar:
        for n in range(1, n_step+1):
            fitness = fun_fitness(group)
            
            best = np.array(group[np.argmin(fitness, axis=0)]) # Elitist Preservation 
            group = domain.mutate(group, p=0.6) # adaptive mutation probability
            group = domain.cross(group, k=math.floor(0.8 * n_city**(1.0-n/n_step))+1, p=0.6)
            # group = domain.cross(group, k=2, p=0.5)
            group[np.argmax(fitness, axis=0)] = best

            pbar.set_postfix({'best_val': fitness.min()})
            pbar.update(1)

        fitness = fun_fitness(group)
        best = group[np.argmin(fitness, axis=0)] # replace the worst
        best_val = fitness.min()
        pbar.set_postfix({'best_val': fitness.min()})
    return best, best_val

def test_tsp():
    tsp = CityTSP(os.path.join(current_dir, 'data', "china.csv"))
    # print(tsp.cal_distance(np.array([[1, 2, 3], [4, 5, 6]])))  # 测试距离矩阵
    domain = PermutationDomain(n=tsp.n, seed=SEED)
    best, best_val = GA(domain, tsp.cal_distance, num=10000, n_step=3000)
    print("Best Path:", best)
    print("Best Distance:", best_val)
    tsp.draw_path(best)

def hard_tsp(size=20):
    tsp = HardCityTSP(os.path.join(current_dir, 'data', "china_main_citys.csv"), size=size)
    # print(tsp.cal_distance(np.array([[1, 2, 3], [4, 5, 6]])))  # 测试距离矩阵
    domain = PermutationDomain(n=tsp.n, seed=SEED)
    best, best_val = GA(domain, tsp.cal_distance, num=size*500, n_step=size*100)
    print("Best Path:", best)
    print("Best Distance:", best_val)
    tsp.draw_path(best) 

if __name__ == '__main__':
    test_tsp()
    # hard_tsp(size=40)