#!/usr/bin/env python3
import subprocess
import json
import os
import shutil
import sys
import argparse

def run_huggingface_download(model_name):
    """Run huggingface-cli download and return the model path."""
    try:
        # Run the huggingface-cli download command
        env = os.environ.copy()
        
        result = subprocess.run(
            ['huggingface-cli', 'download', model_name],
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        
        # The path is typically the last line of output
        model_path = result.stdout.strip().split('\n')[-1]
        print(f"Model downloaded to: {model_path}")
        return model_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def backup_and_modify_tokenizer_config(model_path, revert=False):
    """Backup tokenizer_config.json and remove specified keys."""
    tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
    backup_path = os.path.join(model_path, 'tokenizer_config.json.old')
    
    # Check if tokenizer_config.json exists
    if not os.path.exists(tokenizer_config_path):
        print(f"Warning: tokenizer_config.json not found in {model_path}")
        return
    
    # Create backup
    try:
        # Remove existing backup if it exists
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print(f"Removed existing backup: {backup_path}")
        
        # Create new backup
        shutil.copy2(tokenizer_config_path, backup_path)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        print(f"Attempting to continue without backup...")
        # Don't exit, just warn and continue
    
    # Load and modify the JSON
    try:
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check if added_tokens_decoder exists
        if 'added_tokens_decoder' not in config:
            print("Warning: 'added_tokens_decoder' key not found in tokenizer_config.json")
            return
        
        # Remove the specified keys
        keys_to_remove = ["151667", "151668"]
        removed_keys = []
        
        if revert:
            config['added_tokens_decoder']['151667'] = {
                "content": "<think>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False
            }
            config['added_tokens_decoder']['151668'] = {
                "content": "</think>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False
            }

        else:
            for key in keys_to_remove:
                if key in config['added_tokens_decoder']:
                    del config['added_tokens_decoder'][key]
                    removed_keys.append(key)
        
        if removed_keys:
            print(f"Removed keys from added_tokens_decoder: {removed_keys}")
        elif revert:
            print("Reverted tokenizer config to the original")
        else:
            print("Keys 151667 and 151668 not found in added_tokens_decoder")
        
        # Write the modified config back
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Modified tokenizer_config.json saved")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error modifying tokenizer config: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download HuggingFace model and fix tokenizer config')
    parser.add_argument('--model_name', help='HuggingFace model name (e.g., Qwen/Qwen3-4B-Base)')
    parser.add_argument('--model_path', help='Direct path to already downloaded model directory')
    parser.add_argument('--revert', action='store_true', help='Revert the tokenizer config to the original')
    
    args = parser.parse_args()
    
    if args.model_path:
        # Use existing model path
        model_path = args.model_path
        print(f"Using existing model path: {model_path}")
    elif args.model_name:
        # Download model
        print(f"Downloading model: {args.model_name}")
        model_path = run_huggingface_download(args.model_name)
    else:
        print("Error: Either --model_name or --model_path must be provided")
        sys.exit(1)
    
    print(f"Processing tokenizer config in: {model_path}")
    backup_and_modify_tokenizer_config(model_path, args.revert)
    
    print("Done!")

if __name__ == "__main__":
    main() 