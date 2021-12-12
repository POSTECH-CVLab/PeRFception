
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