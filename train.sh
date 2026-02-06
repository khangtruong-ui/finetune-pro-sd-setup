python train.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --revision="flax" \
    --dataset_name="KhangTruong/NWPU_Split" \
    --resolution=256 --center_crop --random_flip \
    --train_batch_size=64 \
    --num_train_epochs=1000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --revision="flax"\
    --output_dir="./sd-full-finetuned" \
