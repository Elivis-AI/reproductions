import torch 
from torch.utils.data import DataLoader


from dataset import MNIST
from model import MNIST_MoE
from config import MoEConfig


BATCH_SIZE=256 
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

dataset=MNIST(is_train=False) # 数据集
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=10,persistent_workers=True)

config = MoEConfig()
model=MNIST_MoE(config).to(DEVICE)

model.load_state_dict(torch.load('model.pth'))

model.eval()
correct = 0
p_experts = 0
for img, label in dataloader:
    logits, ce, _= model(img.to(DEVICE))
    
    correct+=(logits.cpu().argmax(-1)==label).sum()
    p_experts += ce.cpu() * len(label)

print('正确率:%.2f'%(correct/(len(dataset))*100))
print('专家负载分布:', p_experts / len(dataset))