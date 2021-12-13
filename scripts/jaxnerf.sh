
# LLFF
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/fern \
    --expname fern --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/horns \
    --expname horns --train --eval
python3 -m run --config configs/jaxnerf/llff.yaml --datadir data/llff/room \
    --expname room --train --eval

# BLENDER
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/chair \
    --expname chair --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/lego \
    --expname lego --train --eval
python3 -m run --config configs/jaxnerf/blender.yaml --datadir data/blender/ship \
    --expname ship --train --eval
    