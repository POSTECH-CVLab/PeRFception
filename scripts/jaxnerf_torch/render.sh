
# LLFF
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/fern \
    --expname fern --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/flower \
    --expname flower --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/fortress \
    --expname fortress --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/horns \
    --expname horns --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/leaves \
    --expname leaves --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/orchids \
    --expname orchids --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/room \
    --expname room --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/trex \
    --expname trex --render --model jaxnerf_torch

# LLFF_LARGE
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/room \
    --expname room_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --render --model jaxnerf_torch
    
# BLENDER
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/chair \
    --expname chair --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/drums \
    --expname drums --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/ficus \
    --expname ficus --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/lego \
    --expname lego --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/materials \
    --expname materials --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/mic \
    --expname mic --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/ship \
    --expname ship --render --model jaxnerf_torch
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --render --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --render --model jaxnerf_torch