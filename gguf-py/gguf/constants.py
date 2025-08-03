"""GGUF (GPT-Generated Unified Format) constants and enumerations.

This module defines the core data types and metadata keys used in GGUF files,
including model architectures, tensor types, and quantization formats.
"""

from __future__ import annotations

import sys
from enum import Enum, IntEnum, auto
from typing import Any

# Core GGUF constants
GGUF_MAGIC             = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION           = 3           # Current GGUF format version
GGUF_DEFAULT_ALIGNMENT = 32          # Default memory alignment

#
# constants
#

GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32

#
# metadata keys
#


class Keys:
    class General:
        ARCHITECTURE         = "general.architecture"
        QUANTIZATION_VERSION = "general.quantization_version"
        ALIGNMENT            = "general.alignment"
        NAME                 = "general.name"
        AUTHOR               = "general.author"
        URL                  = "general.url"
        DESCRIPTION          = "general.description"
        LICENSE              = "general.license"
        SOURCE_URL           = "general.source.url"
        SOURCE_HF_REPO       = "general.source.huggingface.repository"
        FILE_TYPE            = "general.file_type"

    class LLM:
        CONTEXT_LENGTH        = "{arch}.context_length"
        EMBEDDING_LENGTH      = "{arch}.embedding_length"
        BLOCK_COUNT           = "{arch}.block_count"
        FEED_FORWARD_LENGTH   = "{arch}.feed_forward_length"
        USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT    = "{arch}.tensor_data_layout"

    class Attention:
        HEAD_COUNT        = "{arch}.attention.head_count"
        HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV         = "{arch}.attention.clamp_kqv"
        LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"

    class Rope:
        DIMENSION_COUNT      = "{arch}.rope.dimension_count"
        FREQ_BASE            = "{arch}.rope.freq_base"
        SCALING_TYPE         = "{arch}.rope.scaling.type"
        SCALING_FACTOR       = "{arch}.rope.scaling.factor"
        SCALING_ORIG_CTX_LEN = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED    = "{arch}.rope.scaling.finetuned"

    class Tokenizer:
        MODEL      = "tokenizer.ggml.model"
        LIST       = "tokenizer.ggml.tokens"
        TOKEN_TYPE = "tokenizer.ggml.token_type"
        SCORES     = "tokenizer.ggml.scores"
        MERGES     = "tokenizer.ggml.merges"
        BOS_ID     = "tokenizer.ggml.bos_token_id"
        EOS_ID     = "tokenizer.ggml.eos_token_id"
        UNK_ID     = "tokenizer.ggml.unknown_token_id"
        SEP_ID     = "tokenizer.ggml.seperator_token_id"
        PAD_ID     = "tokenizer.ggml.padding_token_id"
        ADD_BOS    = "tokenizer.ggml.add_bos_token"
        ADD_EOS    = "tokenizer.ggml.add_eos_token"
        HF_JSON    = "tokenizer.huggingface.json"
        RWKV       = "tokenizer.rwkv.world"
    
    class Optiml:
        SPARSE_THRESHOLD = "Optiml.sparse_threshold"

    class Split:
        VRAM_CAPACITY = "split.vram_capacity"


#
# recommended mapping of model tensor names for storage in gguf
#


class MODEL_ARCH(IntEnum):
    """Supported model architectures in GGUF format."""
    LLAMA     = auto()    # LLaMA architecture (Meta)
    FALCON    = auto()    # Falcon architecture (TII)
    BAICHUAN  = auto()    # Baichuan architecture (Baichuan Intelligence)
    GPT2      = auto()    # GPT-2 architecture (OpenAI)
    GPTJ      = auto()    # GPT-J architecture (EleutherAI)
    GPTNEOX   = auto()    # GPT-NeoX architecture (EleutherAI)
    OPT       = auto()    # OPT architecture (Meta)
    MPT       = auto()    # MPT architecture (MosaicML)
    STARCODER = auto()    # StarCoder architecture (BigCode)
    PERSIMMON = auto()    # Persimmon architecture (ADEPT)
    REFACT    = auto()    # Refact architecture (SmallCloud)
    BERT      = auto()    # BERT architecture (Google)
    BLOOM     = auto()    # BLOOM architecture (BigScience)
    STABLELM  = auto()    # StableLM architecture (Stability AI)
    BAMBOO    = auto()    # Bamboo architecture (BambooAI)


class MODEL_TENSOR(IntEnum):
    """Model tensor types used in GGUF files."""
    TOKEN_EMBD      = auto()  # Token embeddings
    TOKEN_EMBD_NORM = auto()  # Token embeddings normalization
    TOKEN_TYPES     = auto()  # Token type embeddings
    POS_EMBD        = auto()  # Position embeddings
    OUTPUT          = auto()  # Output layer
    OUTPUT_NORM     = auto()  # Output normalization
    ROPE_FREQS      = auto()  # RoPE frequencies
    ATTN_Q          = auto()  # Attention query weights
    ATTN_K          = auto()  # Attention key weights
    ATTN_V          = auto()  # Attention value weights
    ATTN_QKV        = auto()  # Combined QKV attention weights
    ATTN_OUT        = auto()  # Attention output weights
    ATTN_NORM       = auto()  # Attention normalization
    ATTN_NORM_2     = auto()  # Secondary attention normalization
    ATTN_ROT_EMBD   = auto()  # Attention rotary embeddings
    FFN_GATE        = auto()  # Feed-forward gate weights
    FFN_DOWN        = auto()  # Feed-forward down projection
    FFN_UP          = auto()  # Feed-forward up projection
    FFN_NORM        = auto()  # Feed-forward normalization
    ATTN_Q_NORM     = auto()  # Attention query normalization
    ATTN_K_NORM     = auto()  # Attention key normalization
    FFN_DOWN_T      = auto()  # Feed-forward down projection (transposed)
    FC_1            = auto()  # Fully connected layer 1
    FC_2            = auto()  # Fully connected layer 2



MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:          "llama",
    MODEL_ARCH.FALCON:         "falcon",
    MODEL_ARCH.BAICHUAN:       "baichuan",
    MODEL_ARCH.GPT2:           "gpt2",
    MODEL_ARCH.GPTJ:           "gptj",
    MODEL_ARCH.GPTNEOX:        "gptneox",
    MODEL_ARCH.OPT:            "opt",
    MODEL_ARCH.MPT:            "mpt",
    MODEL_ARCH.STARCODER:      "starcoder",
    MODEL_ARCH.PERSIMMON:      "persimmon",
    MODEL_ARCH.REFACT:         "refact",
    MODEL_ARCH.BERT:           "bert",
    MODEL_ARCH.BLOOM:          "bloom",
    MODEL_ARCH.STABLELM:       "stablelm",
    MODEL_ARCH.BAMBOO:         "bamboo",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:      "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM: "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:     "token_types",
    MODEL_TENSOR.POS_EMBD:        "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:     "output_norm",
    MODEL_TENSOR.OUTPUT:          "output",
    MODEL_TENSOR.ROPE_FREQS:      "rope_freqs",
    MODEL_TENSOR.ATTN_NORM:       "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:     "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:        "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:          "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:          "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:          "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:        "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:   "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_Q_NORM:     "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:     "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.FFN_NORM:        "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_GATE:        "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:        "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:          "blk.{bid}.ffn_up",
    MODEL_TENSOR.FFN_DOWN_T:      "blk.{bid}.ffn_down_t",
    MODEL_TENSOR.FC_1:            "blk.{bid}.fc1",
    MODEL_TENSOR.FC_2:            "blk.{bid}.fc2",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_DOWN_T,
        MODEL_TENSOR.FC_1,
        MODEL_TENSOR.FC_2,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STABLELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAMBOO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_DOWN_T,
        MODEL_TENSOR.FC_1,
        MODEL_TENSOR.FC_2,
    ],
    MODEL_ARCH.GPT2: [
        # TODO
    ],
    MODEL_ARCH.OPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.ROPE_FREQS,
    ],
    MODEL_ARCH.BAMBOO: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
}

#
# types
#


class TokenType(IntEnum):
    """Tokenizer token types."""
    NORMAL       = 1  # Regular text token
    UNKNOWN      = 2  # Unknown/undefined token
    CONTROL      = 3  # Control/special token
    USER_DEFINED = 4  # User-defined token
    UNUSED       = 5  # Reserved/unused token type
    BYTE         = 6  # Byte-level token


class RopeScalingType(Enum):
    """RoPE (Rotary Position Embedding) scaling types."""
    NONE   = 'none'    # No scaling
    LINEAR = 'linear'  # Linear scaling
    YARN   = 'yarn'    # YaRN scaling method


class GGMLQuantizationType(IntEnum):
    """GGML tensor quantization types."""
    F32  = 0   # 32-bit floating point
    F16  = 1   # 16-bit floating point
    Q4_0 = 2   # 4-bit integer quantization
    Q4_1 = 3   # 4-bit integer quantization (with scale)
    Q5_0 = 6   # 5-bit integer quantization
    Q5_1 = 7   # 5-bit integer quantization (with scale)
    Q8_0 = 8   # 8-bit integer quantization
    Q8_1 = 9   # 8-bit integer quantization (with scale)
    Q2_K = 10  # 2-bit k-quantization
    Q3_K = 11  # 3-bit k-quantization
    Q4_K = 12  # 4-bit k-quantization
    Q5_K = 13  # 5-bit k-quantization
    Q6_K = 14  # 6-bit k-quantization
    Q8_K = 15  # 8-bit k-quantization
    I8 = 16,   # 8-bit integer
    I16 = 17   # 16-bit integer
    I32 = 18,  # 32-bit integer


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    """GGUF metadata value types used for key-value pairs."""
    UINT8   = 0   # 8-bit unsigned integer
    INT8    = 1   # 8-bit signed integer
    UINT16  = 2   # 16-bit unsigned integer
    INT16   = 3   # 16-bit signed integer
    UINT32  = 4   # 32-bit unsigned integer
    INT32   = 5   # 32-bit signed integer
    FLOAT32 = 6   # 32-bit floating point
    BOOL    = 7   # Boolean value (0=false, 1=true)
    STRING  = 8   # UTF-8 null-terminated string
    ARRAY   = 9   # Homogeneous array (elements share type)
    UINT64  = 10  # 64-bit unsigned integer
    INT64   = 11  # 64-bit signed integer
    FLOAT64 = 12  # 64-bit floating point

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        """Determine the appropriate GGUFValueType for a given Python value.
        
        Args:
            val: Input value to classify
            
        Returns:
            GGUFValueType: The corresponding GGUF value type
            
        Raises:
            SystemExit: If the value type is not recognized
        """
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            print("Unknown type:", type(val))
            sys.exit()


# Quantization block size for K-quant methods
QK_K = 256

# GGML quantization type sizes dictionary
# Format: {GGMLQuantizationType: (block_size, type_size)}
# block_size - number of elements per quant block
# type_size - size in bytes of the quantized type
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32:  (1, 4),       # 32-bit float (no quantization)
    GGMLQuantizationType.F16:  (1, 2),       # 16-bit float (no quantization)
    GGMLQuantizationType.Q4_0: (32, 2 + 16), # 4-bit quantization (type 0)
    GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16), # 4-bit quantization (type 1)
    GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16), # 5-bit quantization (type 0)
    GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16), # 5-bit quantization (type 1)
    GGMLQuantizationType.Q8_0: (32, 2 + 32), # 8-bit quantization (type 0)
    GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32), # 8-bit quantization (type 1)
    GGMLQuantizationType.Q2_K: (256, 2 + 2 + QK_K // 16 + QK_K // 4), # 2-bit k-quant
    GGMLQuantizationType.Q3_K: (256, 2 + QK_K // 4 + QK_K // 8 + 12),   # 3-bit k-quant
    GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),         # 4-bit k-quant
    GGMLQuantizationType.Q5_K: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12), # 5-bit k-quant
    GGMLQuantizationType.Q6_K: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16), # 6-bit k-quant
    GGMLQuantizationType.Q8_K: (256, 4 + QK_K + QK_K // 8),           # 8-bit k-quant
}


# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = Keys.General.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = Keys.General.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = Keys.General.ALIGNMENT
KEY_GENERAL_NAME                 = Keys.General.NAME
KEY_GENERAL_AUTHOR               = Keys.General.AUTHOR
KEY_GENERAL_URL                  = Keys.General.URL
KEY_GENERAL_DESCRIPTION          = Keys.General.DESCRIPTION
KEY_GENERAL_LICENSE              = Keys.General.LICENSE
KEY_GENERAL_SOURCE_URL           = Keys.General.SOURCE_URL
KEY_GENERAL_SOURCE_HF_REPO       = Keys.General.SOURCE_HF_REPO
KEY_GENERAL_FILE_TYPE            = Keys.General.FILE_TYPE

# LLM
KEY_CONTEXT_LENGTH        = Keys.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH      = Keys.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT           = Keys.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH   = Keys.LLM.FEED_FORWARD_LENGTH
KEY_USE_PARALLEL_RESIDUAL = Keys.LLM.USE_PARALLEL_RESIDUAL
KEY_TENSOR_DATA_LAYOUT    = Keys.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT        = Keys.Attention.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV     = Keys.Attention.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS    = Keys.Attention.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV         = Keys.Attention.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS     = Keys.Attention.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS = Keys.Attention.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT      = Keys.Rope.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE            = Keys.Rope.FREQ_BASE
KEY_ROPE_SCALING_TYPE         = Keys.Rope.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR       = Keys.Rope.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN = Keys.Rope.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED    = Keys.Rope.SCALING_FINETUNED

# tokenization
KEY_TOKENIZER_MODEL      = Keys.Tokenizer.MODEL
KEY_TOKENIZER_LIST       = Keys.Tokenizer.LIST
KEY_TOKENIZER_TOKEN_TYPE = Keys.Tokenizer.TOKEN_TYPE
KEY_TOKENIZER_SCORES     = Keys.Tokenizer.SCORES
KEY_TOKENIZER_MERGES     = Keys.Tokenizer.MERGES
KEY_TOKENIZER_BOS_ID     = Keys.Tokenizer.BOS_ID
KEY_TOKENIZER_EOS_ID     = Keys.Tokenizer.EOS_ID
KEY_TOKENIZER_UNK_ID     = Keys.Tokenizer.UNK_ID
KEY_TOKENIZER_SEP_ID     = Keys.Tokenizer.SEP_ID
KEY_TOKENIZER_PAD_ID     = Keys.Tokenizer.PAD_ID
KEY_TOKENIZER_HF_JSON    = Keys.Tokenizer.HF_JSON
KEY_TOKENIZER_RWKV       = Keys.Tokenizer.RWKV
