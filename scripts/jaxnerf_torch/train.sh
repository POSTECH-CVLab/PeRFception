
# LLFF
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fern \
    --expname fern --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/flower \
    --expname flower --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fortress \
    --expname fortress --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/horns \
    --expname horns --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/leaves \
    --expname leaves --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/orchids \
    --expname orchids --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/room \
    --expname room --train 
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/trex \
    --expname trex --train 

# LLFF_LARGE
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/room \
    --expname room_large --train 
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --train 
# BLENDER
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/chair \
    --expname chair --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/drums \
    --expname drums --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ficus \
    --expname ficus --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/lego \
    --expname lego --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/materials \
    --expname materials --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/mic \
    --expname mic --train 
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --train 
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --train 
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --train 