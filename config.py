import argparse

def config_parser():

    parser = argparse.ArgumentParser()

    # model options
    model = parser.add_argument_group("model shared")
    model.add_argument("--model", type=str, help="the running model")

    model.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )

    model.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )    
    model.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    model.add_argument(
        "--batching",
        choices=["single_image", "all_images"], type=str,
        help="strategy to select rays",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument(
        "--max_steps", type=int, default=100000,
        help="number of iterations"
    )

    ray = parser.add_argument_group("rays") 
    ray.add_argument(
        "--use_pixel_centers", action="store_true", default=False,
        help="add a half pixel while generating rays"
    )

    # dataset options
    dataset = parser.add_argument_group("dataset option")
    dataset.add_argument(
        "--dataset_type", type=str, default="llff",
        help="options: llff / blender",
    )
    dataset.add_argument(
        "--testskip", type=int, default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## blender flags
    blender = parser.add_argument_group("blender")
    blender.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    blender.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    llff = parser.add_argument_group("llff flags")
    llff.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    llff.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    llff.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    llff.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    llff.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # metadata: external options for training
    metadata = parser.add_argument_group("metadata")
    metadata.add_argument(
        "--i_print", type=int, default=1000,
        help="frequency of console printout and metric logging",
    )
    metadata.add_argument(
        "--i_weights", type=int, default=50000,
        help="frequency of storing weights"
    )
    metadata.add_argument(
        "--i_validation", type=int, default=50000,
        help="frequency of validation"
    )
    metadata.add_argument(
        "--debug", action="store_true", default=False,
        help="run with debug mode"
    )
    metadata.add_argument(
        "--expname", type=str, required=True, help="experiment name"
    )
    metadata.add_argument(
        "--basedir", type=str, default="./logs/",
        help="where to store ckpts and logs"
    )
    metadata.add_argument(
        "--datadir", type=str, default="./data/llff/fern", 
        help="input data directory"
    )

    runmode = parser.add_argument_group("running mode")
    runmode.add_argument(
        "--train", action="store_true", default=False, help="run with train mode"
    )
    runmode.add_argument(
        "--eval", action="store_true", default=False, help="run with eval mode"
    )
    runmode.add_argument(
        "--bake", action="store_true", default=False, help="bake the trained model"
    )
    runmode.add_argument(
        "--render", action="store_true", default=False, help="render to generate video"
    )
    runmode.add_argument(
        "--skip_validation", action="store_true", default=False, 
    )
    runmode.add_argument(
        "--tpu", action="store_true", default=False, help="run with tpus"
    )
    runmode.add_argument(
        "--tpu_num", type=int, default=0, help="number of tpu"
    )
    runmode.add_argument(
        "--num_workers", type=int, default=8, help="number of workers in dataloaders"
    )    
    runmode.add_argument(
        "--seed", type=int, default=0, help="seed to fix"
    )

    config = parser.add_argument_group("config")
    config.add_argument(
        "--config", type=str, default=None, help="path to config file"
    )

    return parser.parse_args(), parser