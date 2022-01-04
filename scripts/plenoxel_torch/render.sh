# Blender
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/chair/ --expname chair --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/drums/ --expname drums --render
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ficus/ --expname ficus --render 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/hotdog/ --expname hotdog --render 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/lego/ --expname lego --render 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/materials/ --expname materials --render 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/mic/ --expname mic --render 
python3 -m run --config configs/plenoxel_torch/blender.yaml --datadir data/blender/ship/ --expname ship --render 

# LLFF
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fern --expname fern --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/flower --expname flower --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/fortress --expname fortress --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/horns --expname horns --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/leaves --expname leaves --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/orchids --expname orchids --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/room --expname room --render
python3 -m run --config configs/plenoxel_torch/llff.yaml --datadir data/llff/trex --expname trex --render

# Tanks and Temples
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_M60 --expname M60 --render
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Playground --expname Playground --render
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_intermediate_Train --expname Train --render
python3 -m run --config configs/plenoxel_torch/tanks_and_temples.yaml --datadir data/tanks_and_temples/tat_training_Truck --expname Truck --render