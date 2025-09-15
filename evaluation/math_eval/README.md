# Evaluation of Absolute Zero Reasoner (AZR) on Math Benchmarks

### Requirements
You can install the required packages with the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install latex2sympy
cd evaluation/math_eval
uv venv --python 3.10.14
source .venv/bin/activate
uv pip install setuptools wheel build
cd eval
tar -xzvf latex2sympy.tar.gz 
cd latex2sympy
uv pip install -e .
cd ../..
# Install other packages. 
uv pip install -r requirements.txt

# Install flash-attn
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
uv pip install pyarrow==19.0.1
```
> Note the `requirements.txt` doesn't limit packages versions. You can use `freezed_requirements.txt` to install all freezed versions but might include some unused packages. For example, when you install, you may install the latest version of `pyarrow` results in errors. Then you can look into `freezed_requirements.txt` and install the specific version of `pyarrow` as shown above.

### Evaluation

First log into huggingface and download the models to be evaluated (if you have not downloaded them yet):

```bash
cd evaluation/math_eval
source .venv/bin/activate

# Download 3B Coder model
hf download andrewzh/Absolute_Zero_Reasoner-Coder-3b --local-dir-use-symlinks False --local-dir ./models/Absolute_Zero_Reasoner-Coder-3b

# Download 7B Coder model  
hf download andrewzh/Absolute_Zero_Reasoner-Coder-7b --local-dir-use-symlinks False --local-dir ./models/Absolute_Zero_Reasoner-Coder-7b

# Download 7B Base model: Note here it is andrewzh2 instead of andrewzh
hf download andrewzh2/Absolute_Zero_Reasoner-Base-7b --local-dir-use-symlinks False --local-dir ./models/Absolute_Zero_Reasoner-Base-7b

# Download 14B Coder model
hf download andrewzh/Absolute_Zero_Reasoner-Coder-14b --local-dir-use-symlinks False --local-dir ./models/Absolute_Zero_Reasoner-Coder-14b

# Download 14B Base model: Note here it is andrewzh2 instead of andrewzh
hf download andrewzh2/Absolute_Zero_Reasoner-Base-14b --local-dir-use-symlinks False --local-dir ./models/Absolute_Zero_Reasoner-Base-14b

hf download Qwen/Qwen2.5-7B --local-dir-use-symlinks False --local-dir ./models/Qwen2.5-7B
```

Use the following script to evaluate AZR 7b on 6 benchmark with greedy decoding. There is also a `run.sh` script to evaluate all models on all benchmarks.

```bash
# eval AZR
bash eval_math_nodes.sh \
    --run_name azr_base_7b_seed2 \
    --init_model <your absolute path to>/models/Absolute_Zero_Reasoner-Base-7b \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2


# eval Qwen2.5-7B
bash eval_math_nodes.sh \
    --run_name qwen2.5_7b_seed2 \
    --init_model <your absolute path to>/models/Qwen2.5-7B \
    --template qwen-boxed \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24 \
    --n_sampling 512 \
    --just_wandb false \
    --seed 2
```

**Notes:**
- The `--init_model` must be the **absolute path** to your model directory. If you have downloaded them in a different directory, you should change it (be careful wiht "andrewzh" and "andrewzh2" in the path).
- You should change `--template` if you are testing other models. It controls the prompt template used for the evaluation.
- Full list of benchmarks tested: `aime24,aime25,amc23,math500,olympiadbench,minerva_math`. See dataset under `data/` for other possible benchmarks.
- You can change `--benchmarks` to test other benchmarks.
   

## Acknowledgement
The codebase is adapted from [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason), which was based on [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).