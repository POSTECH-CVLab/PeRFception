# Blender
python3 -m run --model plenoxel_torch --datadir data/blender/chair/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/drums/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/ficus/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/hotdog/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/lego/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/materials/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/mic/ --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/blender/ship/ --train --eval --render

# LLFF
python3 -m run --model plenoxel_torch --datadir data/llff/fern --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/flower --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/fortress --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/horns --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/leaves --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/orchids --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/room --train --eval --render
python3 -m run --model plenoxel_torch --datadir data/llff/trex --train --eval --render

# Tanks and Temples
python3 -m run --model plenoxel_torch --model plenoxel_torch --datadir data/tanks_and_temples/tat_intermediate_M60 --train --eval --render
python3 -m run --model plenoxel_torch --model plenoxel_torch --datadir data/tanks_and_temples/tat_intermediate_Playground --train --eval --render
python3 -m run --model plenoxel_torch --model plenoxel_torch --datadir data/tanks_and_temples/tat_intermediate_Train --train --eval --render
python3 -m run --model plenoxel_torch --model plenoxel_torch --datadir data/tanks_and_temples/tat_training_Truck --train --eval --render