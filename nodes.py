import os.path
import dynamicprompts
from dynamicprompts.generators import (
    RandomPromptGenerator,
    FeelingLuckyGenerator
)
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager


NODE_FILE = os.path.abspath(__file__)
NUI_SUITE_ROOT = os.path.dirname(NODE_FILE)


class DynamicPromptsTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "clip": ("CLIP", ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "STRING",)
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

    def encode(self, text, clip, seed):
        prompt_generator = RandomPromptGenerator(
            self._wildcard_manager,
            seed=seed,
            parser_config=self._parser_config,
            unlink_seed_from_prompt=False,
            ignore_whitespace=False
        )

        all_prompts = prompt_generator.generate(text, 1) or [""]
        prompt = all_prompts[0]

        return ([[clip.encode(prompt), {}]], prompt, )


class FeelingLuckyTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "clip": ("CLIP", ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("CONDITIONING","STRING",)
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

    def encode(self, text, clip, seed):
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

        return ([[clip.encode(prompt), {}]], prompt, )


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
    "Nui.DynamicPromptsTextEncode": DynamicPromptsTextEncode,
    "Nui.FeelingLuckyTextEncode": FeelingLuckyTextEncode,
    "Nui.OutputString": OutputString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nui.DynamicPromptsTextEncode": "Dynamic Prompts Text Encode",
    "Nui.FeelingLuckyTextEncode": "Feeling Lucky Text Encode",
    "Nui.OutputString": "Output String",
}
