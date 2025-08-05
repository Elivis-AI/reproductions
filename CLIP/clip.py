import torch 
import torch.nn as nn 
import torch.nn.functional as F

from encoder import ImageEncoder, TextEncoder

class CLIP(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.image_encoder = ImageEncoder(dim)
        self.text_encoder = TextEncoder(dim)

    def forward(self, imgs, texts):
        img_emb = self.image_encoder(imgs)
        text_emb = self.text_encoder(texts)
        return img_emb @ text_emb.T

    def get_loss(self, imgs, texts):
        assert imgs.size(0) == texts.size(0)
        logits = self(imgs, texts)
        targets = torch.arange(len(texts)).to(imgs.device)
        loss_i = F.cross_entropy(logits,targets)
        loss_t = F.cross_entropy(logits.T,targets)
        return (loss_i+loss_t) / 2

    @torch.no_grad()
    def encode_images(self, imgs):
        return self.image_encoder(imgs)

    @torch.no_grad()
    def encode_texts(self, texts):
        return self.text_encoder(texts)   

    @classmethod
    def load_from_checkpoint(cls, file):
        instance = cls()
        instance.load_state_dict(torch.load(file, weights_only=True), strict=True)
        return instance

    def make_vector_db(self, dataset, file, batch_size=1024):
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        dataloader = DataLoader(dataset, batch_size=batch_size)
        device = next(self.parameters()).device
        k_imgs_list, imgs_list = [], []
        k_labels_list, labels_list = [], []
        for imgs, labels in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            k_imgs_list.append(self.encode_images(imgs))
            k_labels_list.append(self.encode_texts(labels))
            imgs_list.append(imgs)
            labels_list.append(labels)
        db = dict(
            k_imgs = torch.cat(k_imgs_list, dim=0).cpu(),
            k_labels = torch.cat(k_labels_list, dim=0).cpu(),
            imgs = torch.cat(imgs_list, dim=0).cpu(),
            labels = torch.cat(labels_list, dim=0).cpu(),      
        )
        torch.save(db, file)
        

class Retriever:
    def __init__(self, db_file, clip_model, device):
        db = torch.load(db_file)
        self.k_imgs = db['k_imgs'].to(device) #[n, 8]
        self.k_labels = db['k_labels'].to(device)
        self.imgs = db['imgs'].to(device) #[n, 8]
        self.labels = db['labels'].to(device)
        self.model = clip_model.to(device)
        self.device = device

    def img2img(self, img, k):
        """top-k retrieval"""
        # assert isinstance(img, torch.Tensor) and img.size(0) == 1
        q = self.model.encode_images(img.to(self.device)) # [1, 8]
        similarities, indices = torch.topk(self.k_imgs @ q.T, k, dim=0) # [n, 1]
        return similarities.squeeze(-1), self.imgs[indices.squeeze(-1)]

    def text2img(self, text, k):
        """top-k retrieval"""
        # assert isinstance(text, torch.Tensor) and text.size(0) == 1
        q = self.model.encode_texts(text.to(self.device)) # [1, 8]
        similarities, indices = torch.topk(self.k_imgs @ q.T, k, dim=0) # [n, 1]
        return similarities.squeeze(-1), self.imgs[indices.squeeze(-1)]        