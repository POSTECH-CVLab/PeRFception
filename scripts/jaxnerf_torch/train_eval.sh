
# LLFF
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fern \
    --expname fern --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/flower \
    --expname flower --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fortress \
    --expname fortress --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/horns \
    --expname horns --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/leaves \
    --expname leaves --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/orchids \
    --expname orchids --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/room \
    --expname room --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/trex \
    --expname trex --train --eval

# LLFF_LARGE
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fern \
    --expname fern_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/flower \
    --expname flower_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/fortress \
    --expname fortress_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/horns \
    --expname horns_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/leaves \
    --expname leaves_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/orchids \
    --expname orchids_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/room \
    --expname room_large --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --datadir data/llff/trex \
    --expname trex_large --train --eval
# BLENDER
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/chair \
    --expname chair --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/drums \
    --expname drums --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ficus \
    --expname ficus --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/hotdog \
    --expname hotdog --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/lego \
    --expname lego --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/materials \
    --expname materials --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/mic \
    --expname mic --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --train --eval
    
# BLENDER LARGE
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/chair \
    --expname chair_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/drums \
    --expname drums_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ficus \
    --expname ficus_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/hotdog \
    --expname hotdog_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/lego \
    --expname lego_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/materials \
    --expname materials_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/mic \
    --expname mic_large --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --datadir data/blender/ship \
    --expname ship_large --train --eval