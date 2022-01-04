# Blender
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/chair/ --expname chair --train
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/drums/ --expname drums --train
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ficus/ --expname ficus --train 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/hotdog/ --expname hotdog --train 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/lego/ --expname lego --train 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/materials/ --expname materials --train 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/mic/ --expname mic --train 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ship/ --expname ship --train 

# LLFF
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fern --expname fern --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/flower --expname flower --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fortress --expname fortress --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/horns --expname horns --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/leaves --expname leaves --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/orchids --expname orchids --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/room --expname room --train 
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/trex --expname trex --train 

# Tanks and Temples
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_M60 --expname M60 --train 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Playground --expname Playground --train 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Train --expname Train --train 
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_training_Truck --expname Truck --train 