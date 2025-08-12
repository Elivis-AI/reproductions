# References
# https://nn.labml.ai/transformers/mlm/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLanguageModel:
    def __init__(
        self,
        vocab_size,
        mask_id,
        no_mask_token_ids=[],
        select_prob=0.15,
        mask_prob=0.8,
        randomize_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.no_mask_token_ids = no_mask_token_ids
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob

        if mask_id not in no_mask_token_ids:
            no_mask_token_ids += [mask_id]

    def _get_mlm_mask(self, gt_token_ids: torch.Tensor) -> torch.Tensor:
        rand_tensor = torch.rand(gt_token_ids.shape, device=gt_token_ids.device)
        no_mask_mask = torch.isin(
            gt_token_ids,
            torch.as_tensor(self.no_mask_token_ids, device=gt_token_ids.device),
        )
        rand_tensor.masked_fill_(mask=no_mask_mask, value=1)

        # "Chooses 15% of the token positions at random for prediction."
        select_mask = (rand_tensor < self.select_prob)
        # `select_mask.sum() / gt_token_ids.numel() ~= 0.15`
        return select_mask

    def _replace_some_tokens(self, token_ids: torch.Tensor, select_mask: torch.Tensor) -> torch.Tensor:
        masked_token_ids = token_ids.clone()

        # "If the $i$-th token is chosen, we replace the $i$-th token with (1) the [MASK] token
        #  80% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        mask_mask = select_mask & (rand_tensor < self.mask_prob)
        # `mask_mask.sum() / select_mask.sum() ~= 0.8`
        masked_token_ids.masked_fill_(mask=mask_mask, value=self.mask_id)

        # "(2) a random token 10% of the time
        #  (3) the unchanged $i$-th token 10% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        randomize_mask = select_mask &\
            (rand_tensor >= self.mask_prob) &\
            (rand_tensor < (self.mask_prob + self.randomize_prob))
        # `randomize_mask.sum() / select_mask.sum() ~= 0.1`
        random_token_ids = torch.randint(
            high=self.vocab_size,
            size=torch.Size((randomize_mask.sum(),)),
            device=masked_token_ids.device,
        )
        masked_token_ids[randomize_mask.nonzero(as_tuple=True)] = random_token_ids
        return masked_token_ids

    def __call__(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        select_mask = self._get_mlm_mask(token_ids)
        masked_token_ids = self._replace_some_tokens(
            token_ids=token_ids, select_mask=select_mask,
        )
        return masked_token_ids, select_mask
    
    def cal_loss(self, pred_token_ids: torch.Tensor, gt_token_ids: torch.Tensor, mlm_mask: torch.Tensor) -> torch.Tensor:
        # gt_token_ids[~mlm_mask] = -100
        gt_token_ids = gt_token_ids.masked_fill(~mlm_mask, -100)
        mlm_loss = F.cross_entropy(pred_token_ids.view(-1, self.vocab_size), gt_token_ids.view(-1))
        return mlm_loss
    

if __name__ == "__main__":
    # Example usage
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    print(tokenizer.vocab_size, tokenizer.model_max_length)
    mlm = MaskedLanguageModel(
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_token_id,
        no_mask_token_ids=[
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        ],
    )

    example_text = "This is an example sentence for masked language modeling."
    output = tokenizer(example_text, return_tensors="pt")
    masked_token_ids, mask = mlm(output.input_ids)
    print(tokenizer.decode(masked_token_ids[0]))
    print(mask)