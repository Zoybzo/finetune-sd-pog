export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="/ssd/sdf/lllrrr/Datasets/POG_sd"
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1

CUDA_VISIABLE_DEVICES=0,3 accelerate launch --gpu_ids="all" --mixed_precision="bf16" --multi_gpu finetune_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pog-model" \
