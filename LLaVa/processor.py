# https://github.com/huggingface/transformers/blob/514b3e81b77ab823d755eb4e3a34b78d1c067454/src/transformers/models/llava/image_processing_llava_fast.py
"""Fast Image processor class for LLaVa."""
from transformers import (
    ProcessorMixin, 
    BatchFeature, 
    CLIPImageProcessor,
    AutoTokenizer
)


class myLlavaProcessor(ProcessorMixin):
    r"""
    Args:
        image_processor ([`CLIPImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`]):
            The tokenizer is a required input.
        patch_size (`int`):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: CLIPImageProcessor,
        tokenizer: AutoTokenizer,
        patch_size: int,
        vision_feature_select_strategy: str = "default",
        chat_template: str = None,
        image_token: str = "<image>",
        num_additional_image_tokens: int = 0,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, padding_text=False, return_tensors="pt") -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")
        
        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            pixel_values = image_inputs["pixel_values"]
            B, C, height, width = pixel_values.shape
            assert B == 1, "TODO: support multi pics"

            num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + self.num_additional_image_tokens
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, padding=padding_text, return_tensors=None)

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
