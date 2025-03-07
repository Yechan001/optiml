# Optiml: Fast Large Language Model Serving with a Consumer-grade GPU

## TL;DR
Optiml is a CPU/GPU LLM inference engine leveraging **activation locality** for your device.

## Demo 🔥

https://github.com/KAIST-KEAI/Optiml/assets/34213478/fe441a42-5fce-448b-a3e5-ea4abb43ba23

Optiml v.s. llama.cpp on a single RTX 4090(24G) running Falcon(ReLU)-40B-FP16 with a 11x speedup!

<sub>Both Optiml and llama.cpp were running on the same hardware and fully utilized VRAM on RTX 4090.</sub>

## Abstract

We introduce Optiml, a high-speed Large Language Model (LLM) inference engine on a personal computer (PC) 
equipped with a single consumer-grade GPU. The key underlying the design of Optiml is exploiting the high **locality** 
inherent in LLM inference, characterized by a power-law distribution in neuron activation. 

This distribution indicates that a small subset of neurons, termed hot neurons, are consistently activated 
across inputs, while the majority, cold neurons, vary based on specific inputs.
Optiml exploits such an insight to design a GPU-CPU hybrid inference engine:
hot-activated neurons are preloaded onto the GPU for fast access, while cold-activated neurons are computed 
on the CPU, thus significantly reducing GPU memory demands and CPU-GPU data transfers.
Optiml further integrates adaptive predictors and neuron-aware sparse operators,
optimizing the efficiency of neuron activation and computational sparsity.

Evaluation shows that Optiml attains an average token generation rate of 13.20 tokens/s, with a peak of 29.08 tokens/s, across various LLMs (including OPT-175B) on a single NVIDIA RTX 4090 GPU,
only 18\% lower than that achieved by a top-tier server-grade A100 GPU.
This significantly outperforms llama.cpp by up to 11.69x while retaining model accuracy.

## Features
Optiml is a high-speed and easy-to-use inference engine for deploying LLMs locally. 

Optiml is fast with:

- **Locality-centric design**: Utilizes sparse activation and 'hot'/'cold' neuron concept for efficient LLM inference, ensuring high speed with lower resource demands.
- **Hybrid CPU/GPU Utilization**: Seamlessly integrates memory/computation capabilities of CPU and GPU for balanced workload and faster processing.

Optiml is flexible and easy to use with:

- **Easy Integration**: Compatible with popular [ReLU-sparse models](https://huggingface.co/SparseLLM) as accurate as their dense counterparts.
- **Local Deployment Ease**: Designed and deeply optimized for local deployment on consumer-grade hardwares, enabling low-latency LLM inference and serving on a single GPU.
- **Backward Compatibility**: While distinct from llama.cpp, you can make use of most of `examples/` the same way as llama.cpp such as server and batched generation. Optiml also supports inference with llama.cpp's model weights for compatibility purpose, but there will be no performance gain.

You can use these models with Optiml today:

- Falcon-40B
- Llama2 family

We have tested Optiml on the following platforms:

- x86-64 CPU (with AVX2 instructions) on Linux
- x86-64 CPU and NVIDIA GPU on Linux
- Apple M Chips on macOS (As we do not optimize for Mac, the performance improvement is not significant now.)

And new features coming soon:

- Mistral-7B model
- Online fine-grained FFN offloading to GPU
- Metal backend for sparse inference on macOS
  
## Getting Started

- [Installation](#setup-and-installation)
- [Model Weights](#model-weights)

## Setup and Installation
### Get the Code

```bash
git clone https://github.com/KAIST-KEAI/Optiml
cd Optiml
```
### Build
In order to build Optiml you have two different options. These commands are supposed to be run from the root directory of the project.

Using `CMake`(3.13+) on Linux or macOS:
* If you have an NVIDIA GPU:
```bash
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```
* If you just CPU:
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Model Weights

Optiml models are stored in a special format called *Optiml GGUF* based on GGUF format, consisting of both LLM weights and predictor weights. You can download Optiml GGUF weights from Hugging Face or convert them from the original model weights and predictor weights.

| Base Model | Optiml GGUF Format | Original Model | Predictor |
|------------|------------------|----------------|---------------------|
| LLaMA(ReLU)-2-7B   | [Optiml/ReluLLaMA-7B-Optiml-GGUF](https://huggingface.co/Optiml/ReluLLaMA-7B-Optiml-GGUF)    | [SparseLLM/ReluLLaMA-7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B)     |  [Optiml/ReluLLaMA-7B-Predictor](https://huggingface.co/Optiml/ReluLLaMA-7B-Predictor)
| LLaMA(ReLU)-2-13B    | [Optiml/ReluLLaMA-13B-Optiml-GGUF](https://huggingface.co/Optiml/ReluLLaMA-13B-Optiml-GGUF)   | [SparseLLM/ReluLLaMA-13B](https://huggingface.co/SparseLLM/ReluLLaMA-13B)  |  [Optiml/ReluLLaMA-13B-Predictor](https://huggingface.co/Optiml/ReluLLaMA-13B-Predictor)
| Falcon(ReLU)-40B    | [Optiml/ReluFalcon-40B-Optiml-GGUF](https://huggingface.co/Optiml/ReluFalcon-40B-Optiml-GGUF)    | [SparseLLM/ReluFalcon-40B](https://huggingface.co/SparseLLM/ReluFalcon-40B)      | [Optiml/ReluFalcon-40B-Predictor](https://huggingface.co/Optiml/ReluFalcon-40B-Predictor)
| LLaMA(ReLU)-2-70B    | [Optiml/ReluLLaMA-70B-Optiml-GGUF](https://huggingface.co/Optiml/ReluLLaMA-70B-Optiml-GGUF)    | [SparseLLM/ReluLLaMA-70B](https://huggingface.co/SparseLLM/ReluLLaMA-70B)      |  [Optiml/ReluLLaMA-70B-Predictor](https://huggingface.co/Optiml/ReluLLaMA-70B-Predictor)

## Inference

For CPU-only and CPU-GPU hybrid inference with all available VRAM, you can use the following instructions to run Optiml:
```bash
  ./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt
```
If you want to limit the VRAM usage of GPU:
```bash
  ./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt --vram-budget $vram_gb
```

As for now, it requires an offline-generated "GPU index" file to split FFNs on GPU. And we found these files are hard to maintain and distribute. We will ship automatic FFN split based on VRAM capacity via [#11](https://github.com/KAIST-KEAI/Optiml/pull/11) very soon.

## Evaluation

![github-eval-4090](https://github.com/KAIST-KEAI/Optiml/assets/34213478/d700fa6c-77ba-462f-a2fc-3fd21c898f33)

![github-eval-2080ti-q4](https://github.com/KAIST-KEAI/Optiml/assets/34213478/0fc1bfc4-aafc-4e82-a865-bec0143aff1a)

Optiml achieves up to 11x and 8x speedup for FP16 and INT4 models!

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of Optiml, supporting Llama-2, Falcon-40B.
- [ ] Support Mistral-7B
- [ ] Support Windows
- [ ] Support text-generation-webui
- [ ] Release perplexity evaluation code
- [ ] Support Metal for Mac
- [ ] Release code for OPT models
- [ ] Release predictor training code 
- [ ] Support online split for FFN network
- [ ] Support Multi-GPU


## Paper and Citation
More technical details can be found in our [paper]().

If you find Optiml useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@techreport{song2023Optiml,
  author      = {Yechan Hwang},
  title       = {Optiml: Fast Large Language Model Serving with a Consumer-grade GPU},
  institution = {KEAI,Korea Advanced Institute of Science and Technology(한국과학기술원)},
  year        = {2025}
}
```

## Acknowledgement
We are thankful for the easily modifiable operator library [ggml](https://github.com/ggerganov/ggml) and execution runtime provided by [llama.cpp](https://github.com/ggerganov/llama.cpp). We also extend our gratitude to [THUNLP](https://nlp.csai.tsinghua.edu.cn/) for their support of ReLU-based sparse models. We also appreciate the research of [Deja Vu](https://proceedings.mlr.press/v202/liu23am.html), which inspires Optiml.
