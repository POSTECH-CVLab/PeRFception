# Blender
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/chair/ --expname chair --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/drums/ --expname drums --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ficus/ --expname ficus --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/hotdog/ --expname hotdog --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/lego/ --expname lego --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/materials/ --expname materials --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/mic/ --expname mic --train --eval --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ship/ --expname ship --train --eval --render

# LLFF
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fern --expname fern --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/flower --expname flower --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fortress --expname fortress --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/horns --expname horns --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/leaves --expname leaves --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/orchids --expname orchids --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/room --expname room --train --eval --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/trex --expname trex --train --eval --render