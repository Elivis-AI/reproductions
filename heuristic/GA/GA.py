import torch
from tqdm import tqdm
from abc import ABC, abstractmethod



class Domain(ABC):
    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def gen(self, num):
        pass


class RealDomain(Domain):
    def __init__(self, ranges, device='cpu'):
        super().__init__(device)
        self.range = torch.tensor(ranges, dtype=torch.float32, device=self.device)
        assert self.range.shape[-1] == 2, "Ranges must have shape [..., 2]"
        self.dim = self.range.shape[:-1]

    def gen(self, num):
        low = self.range[..., 0]
        high = self.range[..., 1]
        rand = torch.rand((num, *self.dim), device=self.device)
        return rand * (high - low) + low

    def select(self, val, fitness):
        prob = fitness / torch.sum(fitness)
        indices = torch.multinomial(prob, num_samples=val.shape[0], replacement=True)
        return val[indices]

    def mutate(self, val, p=0.2):
        mask = torch.rand_like(val) < p
        new_vals = self.gen(val.shape[0])
        return torch.where(mask, new_vals, val)

    def cross(self, val, p=0.8, alpha=0.5):
        num = val.shape[0]
        n = int(p * num)
        if n % 2 != 0:
            n += 1

        id_pairs = torch.randperm(num, device=self.device)[:n]
        pairs = val[id_pairs]
        x, y = pairs[::2], pairs[1::2]
        mask = (torch.rand_like(x) < 0.5)
        delta = (y - x) * mask * (1 - alpha)
        pairs[::2], pairs[1::2] = x + delta, y - delta
        val[id_pairs] = pairs
        return val

class IntDomain(Domain):
    def __init__(self, ranges, device='cpu'):
        super().__init__(device)
        self.range = torch.tensor(ranges, dtype=torch.int32, device=device)
        assert self.range.shape[-1] == 2
        self.dim = self.range.shape[:-1]

    def gen(self, num):
        low = self.range[..., 0]
        high = self.range[..., 1]
        shape = (num, *self.dim)
        return torch.randint_like(torch.empty(shape, dtype=torch.int32, device=self.device), low=low, high=high)

    def select(self, val, fitness):
        prob = fitness / torch.sum(fitness)
        idx = torch.multinomial(prob, num_samples=val.shape[0], replacement=True)
        return val[idx]

    def mutate(self, val, p=0.2):
        mask = torch.rand_like(val.float()) < p
        new_vals = self.gen(val.shape[0])
        return torch.where(mask, new_vals, val)

    def cross(self, val, p=0.8):
        num = val.shape[0]
        n = int(p * num)
        if n % 2 != 0:
            n += 1

        idx = torch.randperm(num, device=self.device)[:n]
        pairs = val[idx]
        x, y = pairs[::2], pairs[1::2]
        mask = (torch.rand_like(x.float()) < 0.5)
        delta = (y - x) * mask
        pairs[::2], pairs[1::2] = x + delta, y - delta
        val[idx] = pairs
        return val

class CategoricalDomain(Domain):
    def __init__(self, categories, device='cpu'):
        super().__init__(device)
        self.categories = torch.tensor(categories, device=self.device)
        self.dim = self.categories.shape[:-1]

    def gen(self, num):
        shape = (num, *self.dim)
        flat_cat = self.categories.view(-1)
        idx = torch.randint(0, len(flat_cat), shape, device=self.device)
        return flat_cat[idx]



class PermutationDomain(Domain):
    def __init__(self, n, device='cpu'):
        self.n = n
        self.device = device

    def gen(self, num):
        # 生成num个随机排列，形状 (num, n)
        perms = []
        for _ in range(num):
            perms.append(torch.randperm(self.n, device=self.device))
        return torch.stack(perms)

    def mutate(self, val, p=0.4):
        assert 0 <= p <= 1
        num = val.shape[0]
        mask = torch.rand(num, device=self.device) < p  # 哪些行参与变异
        out = val.clone()
        rows = torch.nonzero(mask).squeeze(-1)
        if rows.numel() == 0:
            return out

        # 对选中行执行两个位置元素交换
        i = torch.randint(0, self.n, (len(rows),), device=self.device)
        j = torch.randint(0, self.n - 1, (len(rows),), device=self.device)
        j = j + (j >= i)  # 保证 i != j

        out[rows, i], out[rows, j] = out[rows, j].clone(), out[rows, i].clone()
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
            idx = (torch.arange(D, device=self.device)[None, :] - shifts[:, None]) % D
            return a[torch.arange(N, device=self.device)[:, None], idx]

        mask = torch.rand(num, device=self.device) < p
        shift = torch.randint(0, self.n, (num,), device=self.device) * mask.long()
        out = roll_rows(val, shift)
        rows = torch.nonzero(mask).squeeze(-1)
        if rows.numel() > 0:
            out[rows, :k] = out[rows, :k].flip(dims=[1])
        return out


def GA(domain: RealDomain, fun_fitness, num=50, n_step=50):
    group = domain.gen(num)
    with tqdm(total=n_step, desc='GA') as pbar:
        for n in range(1, n_step + 1):
            fitness = fun_fitness(group)
            best_idx = torch.argmin(fitness)
            best = group[best_idx].clone()

            group = domain.select(group, fitness)
            group = domain.cross(group)
            mut_p = torch.rand(1).item() ** (1.0 - n / n_step)
            group = domain.mutate(group, p=mut_p)
            group[best_idx] = best  # Elitist preservation

            pbar.set_postfix({'best_val': fitness.min().item()})
            pbar.update(1)

        fitness = fun_fitness(group)
        best = group[torch.argmin(fitness)]
        best_val = fitness.min().item()
        pbar.set_postfix({'best_val': best_val})
    return best, best_val


if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def schaffer_n4(x):
        x0, x1 = x[..., 0], x[..., 1]
        numerator = torch.cos(torch.sin(torch.abs(x0 ** 2 - x1 ** 2))) ** 2 - 0.5
        denominator = (1.0 + 0.001 * (x0 ** 2 + x1 ** 2)) ** 2
        return 0.5 + numerator / denominator

    res = []
    for i in range(10):
        domain = RealDomain([(-10, 10), (-10, 10)], device=device)
        _, best_val = GA(domain, schaffer_n4, num=100, n_step=500)
        res.append(best_val)

    val = schaffer_n4(torch.tensor([0., 1.25313], device=device)).item()
    val_ga = torch.tensor(res).mean().item()

    print(f'the minimum: {val}')
    print(f'the GA minimum: {val_ga}')
    print('error: {:<3f} %'.format((val_ga / val - 1.0) * 100))
    # the minimum: 0.29257863759994507
    # the GA minimum: 0.2926102578639984
    # error: 0.010807 %