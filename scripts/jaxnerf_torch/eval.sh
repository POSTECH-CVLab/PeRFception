
# LLFF
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fern \
    --expname fern --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/flower \
    --expname flower --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fortress \
    --expname fortress --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/horns \
    --expname horns --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/leaves \
    --expname leaves --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/orchids \
    --expname orchids --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/room \
    --expname room --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/trex \
    --expname trex --eval

# LLFF_LARGE
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/room \
    --expname room_large --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --eval
    
# BLENDER
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/chair \
    --expname chair --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/drums \
    --expname drums --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ficus \
    --expname ficus --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/lego \
    --expname lego --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/materials \
    --expname materials --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/mic \
    --expname mic --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --eval
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --eval