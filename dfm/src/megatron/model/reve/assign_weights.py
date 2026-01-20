import torch
import torch.nn as nn
import os

# Set necessary env vars for MCore
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

import sys
sys.path.append(os.path.abspath("dfm/src/megatron/model/reve/reve_pytorch"))

from megatron.core.transformer.transformer_config import TransformerConfig
from dfm.src.megatron.model.reve.reve_provider import ReveFullModelProvider, ReveSmallModelProvider
from megatron.core import parallel_state, tensor_parallel
from dfm.src.megatron.model.reve.reve_model import ReveModel
from dfm.src.megatron.model.reve.reve_pytorch.model import ReveV2
from dfm.src.megatron.model.reve.reve_pytorch.mock_train_reve import get_full_config, get_small_config

def initialize_megatron():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:29500', rank=0, world_size=1)
    
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        
    # Initialize RNG for Tensor Parallelism
    tensor_parallel.random.model_parallel_cuda_manual_seed(1234)

def main():
    initialize_megatron()
    
    # Config parameters
    simple_config = get_small_config()
    
    # Prepare MCore Config
    # ReveModel expects config to have standard TransformerConfig fields plus the custom ones
    mcore_config = ReveSmallModelProvider(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            sequence_parallel=False,
            seq_length=1024,
            # Calculated fields must be provided explicitly as they are properties/calculated in ReveModel or used there
            kv_channels=simple_config["dims_per_head"],
            num_query_groups=simple_config["num_heads"],
        )
    
    print("Initializing ReveModel (MCore)...")
    reve_mcore = ReveModel(config=mcore_config)
    
    print("Initializing ReveV2 (Simple)...")
    reve_simple = ReveV2(**simple_config)
    
    print("Copying weights from MCore to Simple...")
    copy_weights(reve_mcore, reve_simple)
    
    print("Weights copied successfully.")
    
    # Save the models if needed or just return them
    os.makedirs("synced_checkpoints", exist_ok=True)
    torch.save(reve_mcore.state_dict(), "synced_checkpoints/reve_mcore_init.pt")
    torch.save(reve_simple.state_dict(), "synced_checkpoints/reve_simple_init.pt")
    print("Saved initialized state dicts to 'synced_checkpoints/reve_mcore_init.pt' and 'synced_checkpoints/reve_simple_init.pt'")


def copy_weights(mcore_model, simple_model):
    with torch.no_grad():
        # Embeddings and Input/Output Layers
        copy_linear(mcore_model.img_in, simple_model.img_in)
        # copy_norm(mcore_model.img_in_norm, simple_model.img_in_norm) # RMSNorm stateless
        
        # Time Embeds
        # CosineEmbed has no weights
        copy_mlp_embed(mcore_model.time_mlp_embed, simple_model.time_mlp_embed)
        copy_mlp_embed(mcore_model.conditioning_signal_mlp_embed, simple_model.conditioning_signal_mlp_embed)
        
        # Text Input
        copy_linear(mcore_model.txt_in, simple_model.txt_in)
        # copy_norm(mcore_model.txt_in_norm, simple_model.txt_in_norm) # RMSNorm stateless
        copy_linear(mcore_model.txt_out, simple_model.txt_out)
        
        # Final Layer
        copy_linear(mcore_model.final_layer, simple_model.final_layer)
        
        # Transformer Blocks
        # Text Decoder (Encoder in ReveV2 terms)
        for i, layer_mcore in enumerate(mcore_model.text_decoder.layers):
            layer_simple = simple_model.txt_blocks[i]
            copy_transformer_layer(layer_mcore, layer_simple, is_cross=False)
            
        # Image Decoder
        for i, layer_mcore in enumerate(mcore_model.decoder.layers):
            layer_simple = simple_model.blocks[i]
            copy_transformer_layer(layer_mcore, layer_simple, is_cross=True)

def copy_linear(mcore_linear, simple_linear):
    # MCore Linear usually has weight [out, in]
    # Simple Linear usually has weight [out, in]
    if simple_linear.weight.shape != mcore_linear.weight.shape:
        print(f"Warning: Linear weight shape mismatch: Simple {simple_linear.weight.shape} vs MCore {mcore_linear.weight.shape}")
    simple_linear.weight.copy_(mcore_linear.weight)
    if simple_linear.bias is not None and mcore_linear.bias is not None:
        simple_linear.bias.copy_(mcore_linear.bias)

def copy_mlp_embed(mcore_mlp, simple_mlp):
    copy_linear(mcore_mlp.in_layer, simple_mlp.in_layer)
    copy_linear(mcore_mlp.out_layer, simple_mlp.out_layer)
    # RMSNorm stateless

def copy_transformer_layer(mcore_layer, simple_layer, is_cross):
    # Self Attention
    # MCore: full_self_attention.linear_qkv
    # Simple: self_attn.q, self_attn.kv
    
    # QKV Split
    # MCore linear_qkv weight is [3*h, h] (if not GQA)
    # Simple q [h, h], kv [2*h, h]
    
    qkv = mcore_layer.full_self_attention.linear_qkv.weight
    hidden_size = mcore_layer.config.hidden_size
    
    # Assuming MCore layout is [Q, K, V] concatenated
    q = qkv[:hidden_size, :]
    k = qkv[hidden_size:2*hidden_size, :]
    v = qkv[2*hidden_size:, :]
    
    simple_layer.self_attn.q.weight.copy_(q)
    # Simple KV is [k, v] concatenated output
    simple_layer.self_attn.kv.weight.copy_(torch.cat([k, v], dim=0))
    
    copy_linear(mcore_layer.full_self_attention.linear_proj, simple_layer.self_attn.proj)
    
    if simple_layer.self_attn.do_modulation:
        copy_modulation(mcore_layer.mod_self_attention, simple_layer.self_attn.mod)
        
    copy_gate_residual(mcore_layer.gate_residual_self_attention, simple_layer.self_attn.gate_residual)
    
    # Cross Attention
    if is_cross and simple_layer.do_cross_attn:
        copy_linear(mcore_layer.cross_attention.linear_q, simple_layer.cross_attn.q)
        copy_linear(mcore_layer.cross_attention.linear_kv, simple_layer.cross_attn.kv)
        copy_linear(mcore_layer.cross_attention.linear_proj, simple_layer.cross_attn.proj)
        
        if simple_layer.cross_attn.do_modulation:
            copy_modulation(mcore_layer.mod_cross_attention, simple_layer.cross_attn.mod)
            
        copy_gate_residual(mcore_layer.gate_residual_cross_attention, simple_layer.cross_attn.gate_residual)
        
    # MLP
    copy_linear(mcore_layer.mlp.linear_fc1, simple_layer.mlp.lin1)
    copy_linear(mcore_layer.mlp.linear_fc2, simple_layer.mlp.lin2)
    
    if simple_layer.mlp.do_modulation:
        copy_modulation(mcore_layer.mod_mlp, simple_layer.mlp.mod)
        
    copy_gate_residual(mcore_layer.gate_residual_mlp, simple_layer.mlp.gate_residual)

def copy_modulation(mcore_mod, simple_mod):
    # mcore_mod.lin is ColumnParallelLinear
    # simple_mod.lin is Linear
    copy_linear(mcore_mod.lin, simple_mod.lin)

def copy_gate_residual(mcore_gate, simple_gate):
    # Norm is stateless
    if simple_gate.do_modulation:
        copy_modulation(mcore_gate.modulation, simple_gate.modulation)
    else:
        # Gate Parameter
        simple_gate.gate.data.copy_(mcore_gate.gate.data)

if __name__ == "__main__":
    main()

