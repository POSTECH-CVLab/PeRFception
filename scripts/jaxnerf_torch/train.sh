
# LLFF
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/fern \
    --expname fern --train --model jaxnerf_torch
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/flower \
    --expname flower --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/fortress \
    --expname fortress --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/horns \
    --expname horns --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/leaves \
    --expname leaves --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/orchids \
    --expname orchids --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/room \
    --expname room --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff.yaml --datadir data/llff/trex \
    --expname trex --train --model jaxnerf_torch 

# LLFF_LARGE
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/room \
    --expname room_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --train --model jaxnerf_torch 
# BLENDER
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/chair \
    --expname chair --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/drums \
    --expname drums --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/ficus \
    --expname ficus --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/lego \
    --expname lego --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/materials \
    --expname materials --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf_torch/blender.yaml --datadir data/blender/mic \
    --expname mic --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --train --model jaxnerf_torch 
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --train --model jaxnerf_torch
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --train --model jaxnerf_torch 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --train --model jaxnerf_torch
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --train --model jaxnerf_torch 