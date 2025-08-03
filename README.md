<p align="center">
  <img src='assets/banner.png' width="100%">
</p>

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-blue?style=flat)](#)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=flat)](#)

[![MIT](https://img.shields.io/badge/License-MIT-silver?style=flat-square)](LICENSE)
[![Twitter Follow](https://img.shields.io/badge/Follow-%40optimltech-silver?style=flat-square&logo=x)](https://x.com/optimltech)
[![GitHub](https://img.shields.io/github/stars/NU-QRG/optiml?style=flat-square&logo=github&label=Stars&color=gold)](https://github.com/NU-QRT/optiml)
[![Docker Image Size (tag)](https://img.shields.io/docker/image-size/optiml/optiml/latest?label=Docker%20image)](https://hub.docker.com/r/optiml/optiml)

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

---

**High-speed Large Language Model (LLM) inference on consumer-grade hardware—right on your PC. Large-scale agent deployment is no longer a datacenter privilege.**

OptiML accelerates local inference by exploiting **activation locality**: a compact set of "hot" neurons fire frequently across inputs, while the long tail of "cold" neurons is input-dependent. OptiML places the hot subset on the GPU and schedules the cold subset on the CPU, delivering strong throughput with low VRAM on everyday hardware.

#### OptiML in Action

<div align="center">

https://github.com/user-attachments/assets/78aa5e69-4215-45ef-a4d5-ea9c73918f56

**llama.cpp (left) vs. OptiML (right) on a single RTX 5080 (2.7x speedup!)**

</div>

---

## Highlights

- **Run large models on a PC:** Achieve server-class throughput with one consumer GPU + CPU.
- **Hybrid CPU/GPU execution:** Keep frequently activated ("hot") neurons on the GPU; compute the long tail ("cold") on the CPU.
- **Lower VRAM pressure:** Fit bigger models via quantization and activation-aware placement.
- **Practical & lightweight:** Simple CLI, Python API, and an HTTP demo server for quick local deployment.

---

## Project Motivation

LLMs exhibit **power-law activation locality**: a small, stable subset of neurons accounts for the majority of activations. OptiML identifies this subset and pins it to the GPU for fast reuse, while streaming the less frequent activations on the CPU. This co-design of placement and scheduling balances **latency**, **throughput**, and **memory usage**, enabling large-model serving on commodity PCs.

---

## Supported Models (examples)

Decoder-only transformer families commonly distributed in GGUF or other quantized formats (e.g., LLaMA-style variants). Coverage expands with operator/back-end availability.

---

## Requirements

- A consumer GPU (NVIDIA/AMD/Apple Silicon) with recent drivers/toolkit
- Modern CPU with AVX2 (or Apple Silicon)
- CMake ≥ 3.20, a C/C++ toolchain
- Python 3.9+ (optional, for bindings and scripts)

---

## Quickstart

### 1) Build from source

```bash
git clone https://github.com/NU-QRG/optiml.git
cd optiml
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOPTIML_CUBLAS=ON \        # CUDA (NVIDIA)
  -DOPTIML_METAL=OFF \        # Apple Silicon (toggle as needed)
  -DOPTIML_OPENCL=OFF         # Other GPU backends (toggle as needed)
cmake --build . -j
```

> **Tip:** Toggle the back-ends that match your machine (e.g., set `OPTIML_METAL=ON` on Apple Silicon).

### 2) (Optional) Python bindings

```bash
cd bindings/python
pip install -e .
```

### 3) Prepare a model

OptiML works well with standard GGUF models. If you have original weights, first convert to GGUF, then optionally quantize:

```bash
# Example: quantize a GGUF model to Q4_K
./build/optiml-quantize --input <model path> --output model-q4_k.gguf --type q4_k
```

### 4) Run text generation (CLI)

```bash
./build/optiml-cli --model model-q4_k.gguf --prompt "Explain activation locality in one paragraph." --n-predict 128
```

### 5) Start the HTTP demo server

```bash
./examples/server/optiml-server --model model-q4_k.gguf --host 127.0.0.1 --port 8080
```

Open the provided minimal web UI and chat locally. The server exposes a simple REST API you can call from any client.

---

## How OptiML Works

1. **Measure activation locality** – Identify neurons that are consistently active across inputs.
2. **Partition neurons** – Tag a small "hot" set and a large "cold" set per layer.
3. **Place & cache** – Pin hot neurons and related weights on the GPU; compute cold activations on the CPU.
4. **Hybrid scheduling** – Overlap CPU/GPU compute and data movement; apply quantization to reduce memory and improve throughput.

---

## Quantization

OptiML supports common integer/block quantization schemes (e.g., `Q4_K`, others in GGUF ecosystems) to shrink model size with minimal quality loss. Use `optiml-quantize` and verify trade-offs with perplexity scripts.

---

## Benchmarking

We provide scripts to measure tokens/s, latency, and perplexity across quant levels, sequence lengths, and batch sizes.

```bash
# Throughput / latency
./build/optiml-bench --model <model path> --n-predict 256 --batch 1

# Perplexity
python examples/perplexity/perplexity.py --model <model path> --data <dataset file>
```

Record VRAM/RAM usage, tokens/s, and quality metrics to compare settings on your hardware.

---

## Python API (preview)

You can also build OptiML-accelerated models using our Python APIs. A simple example is provided below. Note that Python bindings are still in an early stage. Expect bugs. If you run into any issue, file a bug report in the repository's issue section.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = "Optiml/Optiml-7B-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# User can directly use the chat interface
# responds, history = model.chat(tokenizer, "Write an article about Artificial Intelligence.", temperature=0.9, top_p=0.9)
# print(responds)
messages = [
    {"role": "user", "content": "Write an article about Artificial Intelligence."},
]
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

model_outputs = model.generate(
    **model_inputs,
    do_sample=True,
    max_new_tokens=1024,
    top_p=0.9,
    temperature=0.9
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

---

## Build Notes

- Enable exactly one GPU back-end that matches your device (`CUBLAS`, `METAL`, `OPENCL`, …).
- For very large models, more VRAM helps, but Optiml’s hybrid placement reduces the requirement.
- Use quantization to lower memory and often improve speed on PC-class hardware.
- Ensure release builds (`-DCMAKE_BUILD_TYPE=Release`) for best performance.

---

## FAQ

**Which models work best?**
Decoder-only transformer families in GGUF with available kernels generally perform well.

**Do I need a high-end GPU?**
Not necessarily. The hybrid layout reduces VRAM pressure by keeping the long tail on the CPU, making consumer GPUs practical.

**How is this different from pure-GPU engines?**
OptiML co-designs placement and scheduling around activation locality, trading a modest amount of CPU work for the ability to serve larger models efficiently on a PC.

---

## Roadmap

- Broader model/operator coverage
- Additional quantization modes and calibration tools
- Auto-tuning for more platforms
- Extended demos (agents, RAG, function calling)

Track progress and propose features via issues/discussions.

---

## Contributing

Contributions are welcome!
When filing issues, include:
- OS/driver/toolkit versions
- CPU/GPU model and RAM/VRAM
- Model/quant settings
- Exact commands and logs

---

## Acknowledgments

OptiML builds on community progress in activation-aware execution, hybrid CPU/GPU scheduling, quantization, and open model formats. Thanks to contributors who make local LLMs fast and accessible.
