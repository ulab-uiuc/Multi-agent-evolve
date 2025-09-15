from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from collections import defaultdict


def main(
    fsdp_checkpoint_path, huggingface_model_path, output_path, pretrained_tokenizer=True, world_size=4
):
    """
    Convert FSDP checkpoint to HuggingFace checkpoint
    Args:
        fsdp_checkpoint_path: path to the FSDP checkpoint
        huggingface_model_path: path to the HuggingFace model
        output_path: path to save the converted checkpoint
    Usage:
        python reason_rl/utils/convert2hf.py \
            checkpoints/azr/azr/test/test_answer/Qwen2.5-7B/answer_conditional/global_step_160_copy/actor \
            checkpoints/azr/azr/test/test_answer/Qwen2.5-7B/answer_conditional/global_step_160_copy/actor/huggingface/ \
            azr_90_composite_160_steps
    """
    state_dict = defaultdict(list)

    for rank in range(int(world_size)):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print("loading", filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)

    # manually change the tokenizer.chat_template to 
    if pretrained_tokenizer:
        chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
        import os
        import json
        with open(os.path.join(output_path, "tokenizer_config.json"), "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["chat_template"] = chat_template
        with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

if __name__ == "__main__":
    fire.Fire(main)
