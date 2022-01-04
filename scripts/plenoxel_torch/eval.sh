# Blender
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/chair/ --expname chair --eval
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/drums/ --expname drums --eval
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ficus/ --expname ficus --eval 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/hotdog/ --expname hotdog --eval 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/lego/ --expname lego --eval 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/materials/ --expname materials --eval 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/mic/ --expname mic --eval 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ship/ --expname ship --eval 

# LLFF
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fern --expname fern --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/flower --expname flower --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fortress --expname fortress --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/horns --expname horns --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/leaves --expname leaves --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/orchids --expname orchids --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/room --expname room --eval 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/trex --expname trex --eval 

# Tanks and Temples
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_M60 --expname M60 --eval 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Playground --expname Playground --eval 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Train --expname Train --eval 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_training_Truck --expname Truck --eval