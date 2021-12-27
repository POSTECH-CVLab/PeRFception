
# LLFF
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fern \
    --expname fern --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/flower \
    --expname flower --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fortress \
    --expname fortress --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/horns \
    --expname horns --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/leaves \
    --expname leaves --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/orchids \
    --expname orchids --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/room \
    --expname room --render
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/trex \
    --expname trex --render

# LLFF_LARGE
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/room \
    --expname room_large --render
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --render
    
# BLENDER
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/chair \
    --expname chair --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/drums \
    --expname drums --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ficus \
    --expname ficus --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/lego \
    --expname lego --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/materials \
    --expname materials --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/mic \
    --expname mic --render
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --render
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --render
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --render