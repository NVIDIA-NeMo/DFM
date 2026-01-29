HF_HOME="/opt/artifacts/" TORCH_HOME="/opt/artifacts/" \
torchrun --nproc-per-node 8 \
    examples/megatron/recipes/wan/vbench_eval/evaluate_vbench.py \
   	--videos_path /opt/artifacts/vbench_outputs/$1/ \
    --output_path /opt/artifacts/vbench_results/$1/ \
    --save_json
