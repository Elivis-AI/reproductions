import torch
from typing import Optional, Tuple, List
from torch import nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List

import warnings
warnings.filterwarnings('ignore')


class LLaVaConfig(PretrainedConfig):
    model_type = "llava-mini"

    def __init__(
        self,
        llm_model: str,
        clip_model: str,
        image_token: str = "<image>", # 注意和数据集对应好
        **kwargs,
    ):
        self.llm_model = llm_model
        self.clip_model = clip_model
        self.image_token = image_token
        self.image_token_id = None
        super().__init__(**kwargs)


class VisionProjector(nn.Module):
    def __init__(self, vision_dim, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(vision_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, image_encoders):
        vision_proj = self.model(image_encoders)
        return vision_proj


class MiniLLaVa(PreTrainedModel):
    config_class = LLaVaConfig

    def __init__(self, config: LLaVaConfig = None):
        super().__init__(config)
        self.image_token_id = config.image_token_id
        assert self.image_token_id is not None, "请先设置image_token_id"

        self.vision_encoder = CLIPVisionModel.from_pretrained(config.clip_model).eval()
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model)
        self.vision_proj = VisionProjector(vision_dim=self.vision_encoder.config.hidden_size, dim=self.llm.config.hidden_size)
        self.embed_tokens = self.llm.model.embed_tokens

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def get_image_embeddings(self, pixel_values: torch.FloatTensor) -> List[torch.Tensor]:
        output = self.vision_encoder(pixel_values=pixel_values)
        vision_embedding = self.vision_proj(output.last_hidden_state[:, 1:, :]) # 去掉[CLS]，(B, N, D)
        return vision_embedding


    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.embed_tokens(
                torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:

        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if pixel_values is not None: # B, C, H, W
            image_features = self.get_image_embeddings(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype) # B, N, D
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            use_cache=use_cache,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs
        )
    

if __name__ == "__main__":
    from transformers import AutoTokenizer, CLIPImageProcessor 
    from processor import myLlavaProcessor

    config = LLaVaConfig(
        llm_model="gpt2",
        clip_model="openai/clip-vit-base-patch16"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, trust_remote_code=True)
    image_processor = CLIPImageProcessor.from_pretrained(config.clip_model)
    if config.image_token not in tokenizer.vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [config.image_token]})
    config.image_token_id = tokenizer.convert_tokens_to_ids(config.image_token)
    processor = myLlavaProcessor(image_processor=image_processor, tokenizer=tokenizer, patch_size=16)


    llavamodel = MiniLLaVa(config)