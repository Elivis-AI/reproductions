import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

class Domain(ABC):
    @abstractmethod
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        pass
    
    @abstractmethod
    def gen(self, num):
        pass


class RealDomain(Domain):
    def __init__(self, ranges, seed=None):
        self.rng = np.random.default_rng(seed)
        self.range = np.array(ranges)
        assert self.range.shape[-1] == 2, "Ranges must be of shape [*D, 2]" 
        self.dim = self.range.shape[:-1]

    def gen(self, num):
        return self.rng.uniform(self.range[..., 0], self.range[..., 1], size=(num, *self.dim))
    
    def select(self, val, fitness):
        fitness = np.array(fitness)
        num = val.shape[0]
        return val[self.rng.choice(num, size=num, p=fitness / np.sum(fitness))]

    def mutate(self, val, p=0.2):
        assert 0 <= p <= 1
        mask = self.rng.random(val.shape) < p
        num = val.shape[0]
        return mask * self.gen(num) + (1 - mask) * val

    def cross(self, val, p=0.8, alpha=0.5):
        assert 0 <= p <= 1
        num = val.shape[0]
        n = int(p * num)
        if n % 2 != 0:
            n += 1

        id_pairs = self.rng.choice(num, size=n, replace=False)
        pairs = val[id_pairs]
        x, y = pairs[::2], pairs[1::2]
        mask = self.rng.random(x.shape) < 0.5
        delta = (y - x) * mask * (1 - alpha)
        pairs[::2], pairs[1::2] = x + delta, y - delta
        val[id_pairs] = pairs

        return val



class IntDomain(Domain):
    def __init__(self, ranges, seed=None):
        self.rng = np.random.default_rng(seed)
        self.range = np.array(ranges)
        assert self.range.shape[-1] == 2, "Ranges must be of shape [*D, 2]" 
        self.dim = self.range.shape[:-1]
    
    def gen(self, num):
        return self.rng.integers(self.range[..., 0], self.range[..., 1], size=(num, *self.dim))

    def select(self, val, fitness):
        fitness = np.array(fitness)
        num = val.shape[0]
        return val[self.rng.choice(num, size=num, p=fitness / np.sum(fitness))]

    def mutate(self, val, p=0.2):
        assert 0 <= p <= 1
        mask = self.rng.random(*val.shape) < p
        num = val.shape[0]
        return mask * self.gen(num) + (1 - mask) * val

    def cross(self, val, p=0.8):
        num = val.shape[0]
        assert 0 <= p <= 1
        n = int(p * num)
        if n % 2 != 0:
            n += 1

        id_pairs = self.rng.choice(num, size=n, replace=False)
        pairs = val[id_pairs]
        x, y = pairs[::2], pairs[1::2]
        mask = self.rng.random(*x.shape) < 0.5
        delta = (y - x) * mask # no interpolation (alpha) for integers
        pairs[::2], pairs[1::2] = x + delta, y - delta
        val[id_pairs] = pairs

        return val
    

class PermutationDomain(Domain):
    def __init__(self, n, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n = n

    def gen(self, num):
        return np.array([self.rng.permutation(self.n) for _ in range(num)])
    
    def mutate(self, val, p=0.4):
        assert 0 <= p <= 1
        num = val.shape[0]
        
        mask = self.rng.random(num) < p  # 哪些行参与变异
        out = val.copy()

        # if np.any(mask):
        rows = np.where(mask)[0]  # 被选中的行的索引 (K,)

        i = self.rng.integers(0, self.n, size=len(rows))
        j = self.rng.integers(0, self.n - 1, size=len(rows))
        j = j + (j >= i)  # 保证 i ≠ j

        out[rows, i], out[rows, j] = out[rows, j], out[rows, i]

        return out
    
    def cross(self, val, k, p=0.4):
        assert 0 <= p <= 1
        num = val.shape[0]
        
        def roll_rows(a, shifts):
            """
            a: (N, D) 数组
            shifts: (N,) 每一行右循环移动多少位
            """
            N, D = a.shape
            idx = (np.arange(D)[None, :] - shifts[:, None]) % D  # shape (N, D)
            return a[np.arange(N)[:, None], idx]

        # if np.any(mask):
        mask = self.rng.random(num) < p
        shift = self.rng.integers(0, self.n) * mask
        out = roll_rows(val, shift)

        rows = np.where(mask)[0]  # 被选中的行的索引 (K,)
        out[rows, :k] = out[rows, :k][:, ::-1]
        return out


def GA(domain: RealDomain, fun_fitness, num=50, n_step=50):
    group = domain.gen(num)
    with tqdm(total=n_step, desc='GA') as pbar:
        for n in range(1, n_step+1):
            fitness = fun_fitness(group)
            
            best = np.array(group[np.argmin(fitness, axis=0)]) # Elitist Preservation 
            group = domain.select(group, fitness)
            group = domain.cross(group)
            group = domain.mutate(group, p=domain.rng.random()**(1.0-n/n_step)) # adaptive mutation probability
            # group[n%num] = best
            group[np.argmin(fitness, axis=0)] = best # replace the worst

            pbar.set_postfix({'best_val': fitness.min()})
            pbar.update(1)

        fitness = fun_fitness(group)
        best = group[np.argmin(fitness, axis=0)]
        best_val = fitness.min()
        pbar.set_postfix({'best_val': fitness.min()})
    return best, best_val



if __name__ == '__main__':
    def schaffer_n4(x):
        x0 = x[..., 0]
        x1 = x[..., 1]
        numerator = (np.cos(np.sin(np.abs(x0**2 - x1**2))))**2 - 0.5
        denominator = (1.0 + 0.001 * (x0**2 + x1**2))**2
        return 0.5 + numerator / denominator


    res = []
    for i in range(10):
        res.append(GA(RealDomain([(-10,10)]*2, seed=42), schaffer_n4, num=100, n_step=500)[1])

    val = schaffer_n4(np.array([0,1.25313])).item()
    val_ga = np.mean(res).item()

    print('the minimum: {0}'.format(val))
    print('the GA minimum: {0}'.format(val_ga))
    print('error: {:<3f} %'.format((val_ga/val-1.0)*100))
    # the minimum: 0.29257863204552975
    # the GA minimum: 0.29257926319380045
    # error: 0.000216 %