import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from GA import PermutationDomain

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TSP:
    def __init__(self, dis_matrix):
        self.dis_matrix = dis_matrix.to(device)

    def cal_distance(self, path):
        # path: (batch_size, n) 或 (n,), int tensor
        if path.dim() == 1:
            path = path.unsqueeze(0)
        path = torch.cat([path, path[:, :1]], dim=1)  # 构造环路
        dist = self.dis_matrix[path[:, :-1], path[:, 1:]]  # shape (batch_size, n)
        return dist.sum(dim=1)


class CityTSP(TSP):
    def __init__(self, csv_path, size=None):
        data = pd.read_csv(csv_path, delimiter=';', header=None).values
        self.pos = data[:size, 1:].astype(float)
        self.dis_matrix = torch.tensor(pairwise_distances(self.pos, metric='euclidean'), dtype=torch.float32, device=device)
        self.city_list = [str(x).strip() for x in data[:size, 0]]
        self.n = len(self.city_list)
        self.best_val = float('inf')
        self.best_path = None

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

        for i, city_idx in enumerate(path):
            ax.text(self.pos[city_idx][0], self.pos[city_idx][1], str(i))

        plt.show()
        
    def dynamic_draw_path(self, path, best_val):
        """
        动态更新当前最优TSP路径显示，仅在最优值改善时调用。
        """
        if best_val >= self.best_val:
            return 
        
        self.best_val = best_val
        self.best_path = path

        plt.ion()  # 打开交互模式
        if not hasattr(self, '_dyn_fig'):
            self._dyn_fig, self._dyn_ax = plt.subplots()
        ax = self._dyn_ax
        ax.cla()  # 清空当前坐标轴内容
        
        ax.set_title("TSP Path  (Distance: {:.2f})".format(best_val))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # 绘制城市点
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c='blue')

        # 绘制路径
        for i in range(len(path)):
            start_city = self.pos[path[i]]
            end_city = self.pos[path[(i + 1) % len(path)]]
            ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], c='red')
        
        # 添加序号标签
        for i, city_idx in enumerate(path):
            ax.text(self.pos[city_idx][0], self.pos[city_idx][1], str(i), fontsize=8, ha='right')
        
        self._dyn_fig.canvas.flush_events()
        plt.pause(0.001)


class HardCityTSP(CityTSP):
    def __init__(self, csv_path, size=None):
        data = pd.read_csv(csv_path, delimiter=',', header=None).values
        max_size = data.shape[0]
        indices = torch.randperm(max_size)[:size].cpu().numpy() if size else range(max_size)

        self.pos = data[indices, -2:].astype(float)
        self.dis_matrix = torch.tensor(pairwise_distances(self.pos, metric='euclidean'), dtype=torch.float32, device=device)
        self.city_list = ['-'.join(x.strip() for x in row) for row in data[indices, :3].astype(str)]
        self.n = len(self.city_list)

        self.best_val = float('inf')
        self.best_path = None


def GA(tsp: CityTSP, num=50, n_step=50, dynamic_draw=False):
    fun_fitness = tsp.cal_distance
    domain = PermutationDomain(n=tsp.n, device=device)
    group = domain.gen(num)
    n_city = domain.n
    dynamic_draw_fn = tsp.dynamic_draw_path 


    with tqdm(total=n_step, desc='GA') as pbar:
        for n in range(1, n_step + 1):
            fitness = fun_fitness(group)
            best_idx = torch.argmin(fitness)
            best = group[best_idx].clone()
            best_val = fitness.min().item()
            pbar.set_postfix({'best_val': best_val})

            if dynamic_draw:
                dynamic_draw_fn(best.clone().cpu().numpy(), best_val)

            group = domain.mutate(group, p=0.6)
            group = domain.cross(group, k=math.floor(0.8 * n_city**(1.0 - n / n_step)) + 1, p=0.6)
            worst_idx = torch.argmax(fitness)
            group[worst_idx] = best

            pbar.update(1)

        if dynamic_draw: plt.ioff()
        # 最后一次更新保证返回最优解
        fitness = fun_fitness(group)
        best_idx = torch.argmin(fitness)
        best = group[best_idx]
        final_best_val = fitness.min().item()
        pbar.set_postfix({'best_val': final_best_val})
    return best.cpu().numpy(), final_best_val


def test_tsp():
    tsp = CityTSP(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "china.csv"))
    best, best_val = GA(tsp, num=10000, n_step=5000, dynamic_draw=True)
    print("Best Path:", best)
    print("Best Distance:", best_val)
    tsp.draw_path(best)


def hard_tsp(size=20):
    tsp = HardCityTSP(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "china_main_citys.csv"), size=size)
    best, best_val = GA(tsp, num=size * 1000, n_step=5 * size**2, dynamic_draw=True)
    print("Best Path:", best)
    print("Best Distance:", best_val)
    tsp.draw_path(best)


if __name__ == '__main__':
    SEED = 42
    torch.manual_seed(SEED)
    # test
    # test_tsp()
    hard_tsp(size=50)
