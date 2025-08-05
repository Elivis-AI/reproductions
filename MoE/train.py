import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import MoEConfig
from dataset import MNIST
from model import MNIST_MoE

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

EPOCH=50
BATCH_SIZE=512

dataset = MNIST()
config = MoEConfig()
model=MNIST_MoE(config).to(DEVICE)
optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)

model.train()

iter_count=0
for epoch in range(EPOCH):
    for imgs,labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, ce, aux_loss = model(imgs)
        
        loss = F.cross_entropy(logits, labels) + aux_loss 
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if iter_count%100==0:
            print('epoch:{} iter:{}, aux_loss: {}, loss:{}'.format(epoch, iter_count, aux_loss.item(), loss.item()))
            torch.save(model.state_dict(),'model.pth')
        iter_count+=1