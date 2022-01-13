import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")
        
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
        "--use_pixel_centers", 
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
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
    dataset.add_argument(
        "--scene_scale", type=float, default=1,
        help="resize the scale of scenes"
    )
    dataset.add_argument(
        "--shuffle_train", 
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="shuffle the train set"
    )

    ## blender flags
    blender = parser.add_argument_group("blender")
    blender.add_argument(
        "--white_bkgd",
        type=str2bool,
        nargs="?",
        const=True,
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )

    ## llff flags
    llff = parser.add_argument_group("llff flags")
    llff.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    llff.add_argument(
        "--no_ndc",
        type=str2bool,
        nargs="?",
        const=True,
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    llff.add_argument(
        "--lindisp",
        type=str2bool,
        nargs="?",
        const=True,
        help="sampling linearly in disparity rather than depth",
    )
    llff.add_argument(
        "--spherify",   
        type=str2bool,
        nargs="?",
        const=True,help="set for spherical 360 scenes"
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
        "--i_validation", type=int, default=50000,
        help="frequency of validation"
    )
    metadata.add_argument(
        "--debug", 
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="run with debug mode"
    )
    metadata.add_argument(
        "--expname", type=str, default=None, help="experiment name"
    )
    metadata.add_argument(
        "--basedir", type=str, default="./logs/",
        help="where to store ckpts and logs"
    )
    metadata.add_argument(
        "--datadir", type=str, default="./data/llff/fern", 
        help="input data directory"
    )
    metadata.add_argument(
        "--postfix", type=str, default="",
        help="post fix to add behind the expname"
    )

    runmode = parser.add_argument_group("running mode")
    runmode.add_argument(
        "--train",  
        type=str2bool,
        nargs="?",
        const=True, 
        default=False, 
        help="run with train mode"
    )
    runmode.add_argument(
        "--eval",    
        type=str2bool,
        nargs="?",
        const=True,
        default=False, 
        help="run with eval mode"
    )
    runmode.add_argument(
        "--bake",         
        type=str2bool,
        nargs="?",
        const=True,
        default=False, 
        help="bake the trained model"
    )
    runmode.add_argument(
        "--render", 
        type=str2bool,
        nargs="?",
        const=True,
        default=False, help="render to generate video"
    )
    runmode.add_argument(
        "--tpu", 
        type=str2bool,
        nargs="?",
        const=True,
        default=False, help="run with tpus"
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
    runmode.add_argument(
        "--use_custom_optim",         
        type=str2bool,
        nargs="?",
        const=True,
        default=False, 
        help="Run with a custom optimization step"
    )
    runmode.add_argument(
        "--run_large_model",         
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="For wandb sweep: run large nerf model"
    )

    config = parser.add_argument_group("config")
    config.add_argument(
        "--config", type=str, default=None, help="path to config file"
    )

    return parser.parse_known_args()[0], parser