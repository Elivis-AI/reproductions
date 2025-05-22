class EMALoss():
    def __init__(self, alpha):
        super().__init__()
        # 更新权重
        self.alpha = alpha
        self.loss = None

    def reset(self):
        self.loss = None

    def update(self, loss):
        if self.loss is None:
            self.loss = loss.item()
        else:
            self.loss = self.alpha * loss.item() + (1 - self.alpha) * self.loss
    
    def item(self):
        return self.loss