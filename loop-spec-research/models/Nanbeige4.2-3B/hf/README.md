---
license: apache-2.0
language:
- en
- zh
library_name: transformers
pipeline_tag: text-generation
tags:
- llm
- nanbeige
base_model:
- Nanbeige/Nanbeige4.2-3B-Base
---

<div align="center">
<img src="figures/nbg.png" width="220" alt="Nanbeige Logo">
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2607.22083"><b>Technical Report</b>👁️</a>
</p>

# News

- 🎉 Nanbeige4.2-3B has taken the top spot in [Artificial Analysis's latest leaderboard for small models](https://artificialanalysis.ai/articles/mobile-phone-intelligence-inference).

# <span id="Introduction">1. Introduction</span>

Nanbeige4.2-3B is a compact agentic model built on [Nanbeige4.2-3B-Base](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base), designed to combine strong agentic behavior with broad reasoning and alignment capabilities. Its Looped Transformer architecture reuses the transformer layers to increase model capacity without adding parameters. With only 3B non-embedding parameters, the model delivers solid performance on general-agent and code-agent tasks.

During supervised fine-tuning (SFT), we expand the diversity of training environments through real-world environment integrations and large-scale environment synthesis. We further diversify task types, task assets, and the agentic scaffolds used for each task. To ensure training data quality, we apply filtering at both the trajectory and turn levels, combining test-case-based validation with rubric-based assessment. During reinforcement learning (RL), we combine outcome and process rewards to improve training stability for the compact model.

<div align="center">

<img src="figures/model_performance.png">
</div>

Key strengths include:

- **Solid Agentic Behavior at the 3B Scale**: Across complex tool-use, office-agent, and code-agent benchmarks, Nanbeige4.2-3B outperforms larger models such as Qwen3.5-9B and Gemma4-12B.
  
- **Strong Reasoning Capabilities**: Nanbeige4.2-3B leads open-source models of comparable size across mathematical, coding, and scientific reasoning tasks, continuing the strong reasoning performance of [Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B).
 
- **Local Personal Assistant**: When integrated with an agentic scaffold designed for personal workflows (e.g., OpenClaw), Nanbeige4.2-3B can support extended tasks spanning daily assistance, office work, and deep research.

> The accompanying [`modeling_nanbeige.py`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/modeling_nanbeige.py) also includes our latest architectural improvements, including **LoopSplit**, **mHC with depth attention**, and **concatenated n-gram embeddings**. These features have been incorporated into Nanbeige4.5, whose training is underway for release later in 2026.

# <span id="Model-Performance">2. Model Performance</span>

## General and Agentic Capabilities

We compare Nanbeige4.2-3B with Qwen3.5 and Gemma4 models across a diverse benchmark suite covering general agents, code agents, reasoning, and alignment capabilities.

<table>
  <thead>
    <tr>
      <th>-</th>
      <th>-</th>
      <th>Nanbeige4.2-3B<sup>1</sup></th>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.5-4B</th>
      <th>Gemma4-12B</th>
      <th>Gemma4-E4B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Parameters</strong></td>
      <td>Total Params</td><td>4B</td><td>10B</td><td>5B</td><td>12B</td><td>8B</td>
    </tr>
    <tr><td>Non-embedding Params</td><td>3B</td><td>8B</td><td>4B</td><td>10B</td><td>4B</td></tr>
    <tr>
      <td rowspan="7"><strong>General Agent</strong><sup>2</sup></td>
      <td>GDPval rubrics</td><td><strong>74.3</strong></td><td>61.9</td><td>46.7</td><td><u>68.5</u></td><td>31.5</td>
    </tr>
      <tr><td>Agent-IF-Oneday</td><td><strong>67.5</strong></td><td><u>60.4</u></td><td>56.9</td><td>—</td><td>—</td></tr>
      <tr><td>Office-QA-Pro<sup>3</sup></td><td><strong>21.1</strong></td><td><u>15.8</u></td><td>8.3</td><td>15.3</td><td>3.1</td></tr>
    <tr><td>Pinch-Bench-V2</td><td><strong>74.7</strong></td><td><u>68.2</u></td><td>63.9</td><td>53.8</td><td>33.3</td></tr>
    <tr><td>Claw-Gym</td><td><strong>65.0</strong></td><td><u>56.1</u></td><td>53.0</td><td>40.8</td><td>16.4</td></tr>
    <tr><td>Claw-Eval<sub>pass^3</sub></td><td><strong>52.2</strong></td><td><u>47.1</u></td><td>36.9</td><td>25.5</td><td>15.9</td></tr>
    <tr><td>MCP-Atlas</td><td><strong>57.8</strong></td><td><u>47.4</u></td><td>40.8</td><td>30.5</td><td>15.0</td></tr>
    <tr>
      <td rowspan="3"><strong>Code Agent</strong><sup>4</sup></td>
      <td>SWE-Bench Verified</td><td><strong>63.6</strong></td><td><u>53.1</u></td><td>38.8</td><td>44.2</td><td>14.0</td>
    </tr>
    <tr><td>SWE-Bench Pro</td><td><strong>46.9</strong></td><td><u>33.8</u></td><td>29.4</td><td>21.9</td><td>4.0</td></tr>
    <tr><td>Terminal-Bench 2.0</td><td><strong>44.1</strong></td><td><u>29.2</u></td><td>25.8</td><td>21.1</td><td>12.4</td></tr>
    <tr>
      <td rowspan="6"><strong>Reasoning</strong></td>
      <td>HLE w/o Search</td><td><strong>17.8</strong></td><td>12.5</td><td>6.8</td><td><u>14.8</u></td><td>4.0</td>
    </tr>
    <tr><td>SciCode</td><td><u>35.6</u></td><td>32.7</td><td>22.7</td><td><strong>38.2</strong></td><td>24.9</td></tr>
    <tr><td>GPQA-Diamond</td><td><strong>87.4</strong></td><td><u>81.7</u></td><td>78.2</td><td>78.8</td><td>60.6</td></tr>
    <tr><td>HMMT-Feb-2026</td><td><strong>82.8</strong></td><td><u>69.6</u></td><td>60.6</td><td>51.5</td><td>24.2</td></tr>
    <tr><td>IMO-Answer-Bench</td><td><strong>67.3</strong></td><td><u>56.3</u></td><td>46.8</td><td>54.5</td><td>24.0</td></tr>
    <tr><td>LiveCodeBench-V6</td><td><strong>72.5</strong></td><td>65.6</td><td>55.8</td><td><u>72.0</u></td><td>55.3</td></tr>
    <tr>
      <td rowspan="3"><strong>Alignment</strong></td>
      <td>AA-LCR</td><td><strong>58.7</strong></td><td><u>58.0</u></td><td>52.0</td><td>55.3</td><td>30.7</td>
    </tr>
    <tr><td>IF-Bench</td><td><u>54.6</u></td><td>54.1</td><td>41.4</td><td><strong>73.5</strong></td><td>44.0</td></tr>
    <tr><td>Recruit-Bench<sup>5</sup></td><td><u>63.3</u></td><td>59.0</td><td>40.7</td><td><strong>69.4</strong></td><td>57.9</td></tr>
  </tbody>
</table>

<small>
<sup>1</sup> All evaluations are conducted in thinking mode with <code>preserve_thinking=true</code> in the chat template.<br>
<sup>2</sup> Office and co-work tasks such as GDPval, Office-QA-Pro, and Agent-IF-Oneday are evaluated with our in-house scaffold.<br>
<sup>3</sup> For OfficeQA-pro, we follow the most challenging evaluation setup: for each question, we provide all the original PDF materials without giving any hints about the relevant documents.<br>
<sup>4</sup> SWE-Bench Verified uses the OpenHands scaffold, SWE-Bench Pro uses the SWE-agent scaffold, and Terminal-Bench 2.0 uses the Terminus 2 scaffold.<br>
<sup>5</sup> Recruit-Bench is our in-house benchmark covering enterprise hiring scenarios (B2C) and job-seeking scenarios for candidates (C2B).<br>
</small>


The results demonstrate that Nanbeige4.2-3B delivers strong performance well beyond its parameter scale. With only 3B non-embedding parameters, it consistently outperforms larger models, including Qwen3.5-9B and Gemma4-12B, across general-agent, code-agent, and reasoning benchmarks, while remaining competitive on alignment tasks.

## Local Personal Assistant

With only 3B non-embedding parameters, Nanbeige4.2-3B is compact enough for local deployment while retaining the agentic capabilities needed for multi-step workflows, making it a natural fit for local personal-assistant applications. To assess this use case in a practical and consistent agent environment, we use OpenClaw, a general-purpose framework that supports daily assistance, office workflows, and deep research tasks. All compared models use the same framework and are evaluated on tasks requiring multi-step interaction with tools and external resources.

<table>
  <thead>
    <tr>
      <th>Capability</th>
      <th>Benchmark</th>
      <th>Nanbeige4.2-3B</th>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.5-4B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Daily Tasks</strong></td>
      <td>Pinch-Bench-V2</td><td><strong>74.7</strong></td><td><u>68.2</u></td><td>63.9</td>
    </tr>
    <tr><td>Claw-Gym</td><td><strong>65.0</strong></td><td><u>56.1</u></td><td>53.0</td></tr>
    <tr>
      <td rowspan="2"><strong>Office Tasks</strong></td>
      <td>GDPval</td><td><strong>68.8</strong></td><td><u>38.0</u></td><td>37.0</td>
    </tr>
    <tr><td>Agent-IF-Oneday</td><td><strong>58.9</strong></td><td><u>32.1</u></td><td>27.0</td></tr>
    <tr>
      <td rowspan="2"><strong>Deep Research</strong></td>
      <td>DeepResearch Bench II</td><td><strong>33.4</strong></td><td><u>28.3</u></td><td>26.0</td>
    </tr>
    <tr><td>ResearchRubrics</td><td><strong>44.8</strong></td><td><u>37.2</u></td><td>35.1</td></tr>
  </tbody>
</table>

Across all six benchmarks, Nanbeige4.2-3B outperforms both Qwen3.5-4B and the larger Qwen3.5-9B. These results support its use as a compact local personal assistant.

# <span id="Quickstart">3. Quickstart</span>

The tokenizer provides a configurable chat template for reasoning and tool-use scenarios:

- `enable_thinking` controls whether the model generates reasoning for the current response. It is enabled by default; set it to `False` for non-thinking mode.
- `preserve_thinking` controls whether reasoning from previous assistant turns is retained in a multi-turn conversation. We recommend `False` for general chat and question answering, and `True` for multi-turn tool use, office tasks, and code-agent workflows.
- Passing `tools` enables the tool-use template. We recommend `tool_call_format="xml"` for the best tool-calling performance; `json` is also supported for compatibility.

The model supports a context length of up to 262,144 tokens (256K).

We recommend adjusting the inference settings according to the target scenario:

| Scenario | Temperature | Max New Tokens |
|---|---:|---:|
| Agentic and tool-use tasks | 1.0 | 65,536 |
| Reasoning and chat tasks | 0.6 | 131,072 |

## Huggingface

Install the required version with `pip install transformers==4.45.1`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Nanbeige/Nanbeige4.2-3B"

tokenizer = AutoTokenizer.from_pretrained(
  model_id,
  use_fast=False,
  trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype="auto",
  device_map="auto",
  trust_remote_code=True
)

messages = [
  {"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}
]
prompt = tokenizer.apply_chat_template(
  messages,
  add_generation_prompt=True,
  tokenize=False
)
input_ids = tokenizer(
  prompt,
  add_special_tokens=False,
  return_tensors="pt"
).input_ids
output_ids = model.generate(
  input_ids.to("cuda"),
  max_new_tokens=131072,
  temperature=0.6,
  top_p=0.95,
  top_k=20,
  eos_token_id=166101
)
response = tokenizer.decode(
  output_ids[0][len(input_ids[0]):],
  skip_special_tokens=True
)
print(response)
```

## SGLang
### Installation
```bash
# Clone repository
git clone -b nbg42 https://github.com/Nanbeige/sglang.git
cd sglang
pip install -e "python"
```
### Usage
```bash
MODEL_PATH=/path/to/your/Nanbeige4.2-3B
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.8  \
    --reasoning-parser nanbeige \
    --tool-call-parser nanbeige
```


## vLLM
### Installation
```bash
# Clone repository
git clone -b nanbeige42 https://github.com/Nanbeige/vllm.git
cd vllm
pip install -e .
```
### Usage
```bash
MODEL_PATH=/path/to/your/Nanbeige4.2-3B
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8  \
    --enable-auto-tool-choice \
    --tool-call-parser nanbeige \
    --reasoning-parser nanbeige
```

## llama.cpp
### Installation
```bash
# Clone repository
git clone -b nanbeige42 https://github.com/Nanbeige/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```
### Usage
```bash
# input: HuggingFace model directory
MODEL_PATH_HF=/path/to/your/Nanbeige4.2-3B
# output: converted / quantized GGUF
MODEL_PATH_BF16_GGUF=/path/to/your/Nanbeige4.2-3B-BF16.gguf
MODEL_PATH_GGUF_Q4_K_M=/path/to/your/Nanbeige4.2-3B-Q4_K_M.gguf

# convert to gguf
python3 convert_hf_to_gguf.py ${MODEL_PATH_HF} \
  --outfile ${MODEL_PATH_BF16_GGUF} \
  --outtype bf16

# quantize to int4
./build/bin/llama-quantize \
  ${MODEL_PATH_BF16_GGUF} \
  ${MODEL_PATH_GGUF_Q4_K_M} \
  Q4_K_M

# run inference
./build/bin/llama-cli \
  -m ${MODEL_PATH_GGUF_Q4_K_M} \
  -ngl 99
```

If you want to use LM Studio: the bundled `llama-server` does not support `nanbeige` yet, so do these two steps:

1. Put the GGUF under `~/.lmstudio/models/<org>/<name>/`
2. Copy this [llama.cpp](#llamacpp) build outputs into the LM Studio backend:
   `${LLAMA_CPP_PATH}/build/bin/*` -> `BACKEND` (example: `~/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.25.2`)

## ollama
### Requirements
- **Go** 
- **llama.cpp** (see the [llama.cpp](#llamacpp) section above)

### Installation
```bash
# Clone repository
# For llama-server (GGUF) backend only, the official ollama repo also works:
# git clone https://github.com/ollama/ollama.git
git clone -b nanbeige42 https://github.com/Nanbeige/ollama.git
cd ollama

# Full native build (MLX Metal on macOS arm64; llama-server payload included)
# Pick one:
cmake -B build .
cmake --build build --parallel $(sysctl -n hw.ncpu)   # macOS
# cmake --build build --parallel $(nproc)             # Linux

# Copy the llama.cpp build output from the section above into Ollama's runtime payload dir
LLAMA_CPP_PATH=/path/to/your/llama.cpp
cp -r ${LLAMA_CPP_PATH}/build/bin/* build/lib/ollama/

go build .
```

Ollama supports two local inference backends:

| Path | Model format | Backend | Typical use |
|------|--------------|---------|-------------|
| **llama-server** | GGUF | llama.cpp (Metal / CUDA / ...) | Traditional GGUF quantized deployment |
| **MLX** | HuggingFace safetensors | MLX (Apple Metal, etc.) | Run BF16 or quantized safetensors directly |


### Usage — llama-server (GGUF)
```bash
# Option 1: pull a published model (no Modelfile on the client)
./ollama serve
./ollama run nanbeige/nanbeige4.2:3b-Q4_K_M
```

```bash
# Option 2: create from a local .gguf produced by llama.cpp
# (see convert_hf_to_gguf.py / llama-quantize above)
MODEL_PATH_GGUF_Q4_K_M=/path/to/your/Nanbeige4.2-3B-Q4_K_M.gguf

# Modelfile
cat > Modelfile <<EOF
FROM ${MODEL_PATH_GGUF_Q4_K_M}
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
EOF

./ollama serve
./ollama create nanbeige42-local -f Modelfile
./ollama run nanbeige42-local
```

### Usage — MLX (safetensors)
```bash
# HuggingFace safetensors directory (config.json + *.safetensors), used directly.
# MLX Metal is for macOS arm64; Linux MLX needs a CUDA MLX backend build.
MODEL_PATH=/path/to/your/Nanbeige4.2-3B

# Modelfile
cat > Modelfile <<EOF
FROM ${MODEL_PATH}
RENDERER nanbeige
PARSER nanbeige
EOF

# Start server (from repo root so it finds build/lib/ollama)
./ollama serve

# Import safetensors model (requires --experimental)
./ollama create nanbeige42-mlx -f Modelfile --experimental

# Optional: quantize on import (int4 / int8 / mxfp4 / mxfp8 / nvfp4)
./ollama create nanbeige42-mlx-q4 -f Modelfile --experimental --quantize int4

# Run
./ollama run nanbeige42-mlx-q4
```



# <span id="Limitations">4. Limitations</span>

While we place great emphasis on model safety throughout the training process, the model may still generate unexpected or inappropriate outputs due to its probabilistic nature. Such outputs may include inaccurate information, bias, discrimination, or other harmful content. Please do not propagate such content. We do not assume responsibility for consequences of disseminating inappropriate information.

# <span id="Citation">5. Citation</span>

If you find our model useful or would like to use it in your own work, please cite as follows:

```bibtex
@article{lab2026nanbeige4,
  title={Nanbeige4. 2-3B: Unlocking Agentic Capabilities in a Compact Mode},
  author={Lab, Nanbeige and Yang, Chen and Huang, Chengrui and Lan, Fufeng and Chen, Hanhui and Zhou, Hao and Song, Huatong and Cao, Jiaqi and Zhu, Jiaying and Niu, Jinlin and others},
  journal={arXiv preprint arXiv:2607.22083},
  year={2026}
}
```

# <span id="Contact">6. Contact</span>

If you have any questions, please open an issue in this repository or contact us at nanbeige@kanzhun.com.
