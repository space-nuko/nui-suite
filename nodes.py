import os.path
import dynamicprompts
from dynamicprompts.generators import (
    RandomPromptGenerator,
    FeelingLuckyGenerator
)
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
import comfy.samplers

from .clip_guidance import common_ksampler


NODE_FILE = os.path.abspath(__file__)
NUI_SUITE_ROOT = os.path.dirname(NODE_FILE)


class DynamicPromptsTextGen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def __init__(self):
        wildcard_dir = os.path.join(NUI_SUITE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start="{",
            variant_end="}",
            wildcard_wrap="__"
        )

    def encode(self, text, seed):
        prompt_generator = RandomPromptGenerator(
            self._wildcard_manager,
            seed=seed,
            parser_config=self._parser_config,
            unlink_seed_from_prompt=False,
            ignore_whitespace=False
        )

        all_prompts = prompt_generator.generate(text, 1) or [""]
        prompt = all_prompts[0]

        return (prompt, )


class FeelingLuckyTextGen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def __init__(self):
        wildcard_dir = os.path.join(NUI_SUITE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start="{",
            variant_end="}",
            wildcard_wrap="__"
        )

    def encode(self, text, seed):
        inner_generator = RandomPromptGenerator(
            self._wildcard_manager,
            seed=seed,
            parser_config=self._parser_config,
            unlink_seed_from_prompt=False,
            ignore_whitespace=False
        )
        prompt_generator = FeelingLuckyGenerator(inner_generator)

        all_prompts = prompt_generator.generate(text, 1) or [""]
        prompt = all_prompts[0]

        return (prompt, )


class ClipGuidedKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "vae": ("VAE", ),
                    "clip": ("CLIP", ),
                    "clip_vision": ("CLIP_VISION", ),
                    "clip_prompt": ("STRING", {"multiline": True}),
                    "clip_scale": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, clip, clip_vision, clip_prompt, clip_scale, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, clip, clip_vision, clip_prompt, clip_scale, denoise=denoise)


class OutputString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "output_string"

    OUTPUT_NODE = True

    CATEGORY = "utils"

    def output_string(self, string):
        return { "ui": { "string": string } }


NODE_CLASS_MAPPINGS = {
    "Nui.DynamicPromptsTextGen": DynamicPromptsTextGen,
    "Nui.FeelingLuckyTextGen": FeelingLuckyTextGen,
    "Nui.ClipGuidedKSampler": ClipGuidedKSampler,
    "Nui.OutputString": OutputString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nui.DynamicPromptsTextEncode": "Dynamic Prompts Text Generator",
    "Nui.FeelingLuckyTextEncode": "Feeling Lucky Text Generator",
    "Nui.ClipGuidedKSampler": "CLIP Guided KSampler",
    "Nui.OutputString": "Output String",
}
