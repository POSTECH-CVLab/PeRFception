import configargparse

def config_parser():

    parser = configargparse.ArgumentParser()

    # model options
    model = parser.add_argument_group("model shared")
    model.add_argument("--model", type=str, help="the running model")
    model.add_argument("--netdepth", type=int, default=8, help="layers in network")
    model.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    model.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    model.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
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
    model.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    model.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    rendering = parser.add_argument_group("rendering")
    rendering.add_argument(
        "--num_coarse_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    rendering.add_argument(
        "--num_fine_samples",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    rendering.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    rendering.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    rendering.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    rendering.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    rendering.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    rendering.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    rendering.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    rendering.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    rendering.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options
    train = parser.add_argument_group("train")
    train.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    train.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    train.add_argument(
        "--lr_init", type=float, default=2.0e-4,
        help="initial learning rate"
    )
    train.add_argument(
        "--lr_final", type=float, default=2.5e-6, 
        help="the final learning rate"
    )
    train.add_argument(
        "--max_steps", type=int, default=1000000,
        help="the maximum number of steps"
    )
    train.add_argument(
        "--lr_delay_steps", type=int, default=2500, 
        help="learning rate delay step"
    )
    train.add_argument(
        "--lr_delay_mult", type=float, default=0.1,
        help="delay factor"
    )
    train.add_argument(
        "--seed", type=int, default=0, help="seed to fix"
    )


    # dataset options
    dataset = parser.add_argument_group("dataset option")
    dataset.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    dataset.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    dvx = parser.add_argument_group("deepvoxel")
    dvx.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
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

    # ray option
    ray = parser.add_argument_group("rays") 
    ray.add_argument(
        "--use_pixel_centers", action="store_true", default=False,
        help="add a half pixel while generating rays"
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
        "--config", is_config_file=True, help="config file path"
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

    snerg = parser.add_argument_group("snerg specific args")
    snerg.add_argument(
        "--num_view_dir_channels", type=int, default=4, 
        help="dimension of specular features"
    )
    snerg.add_argument(
        "--viewdir_netdepth", type=int, default=2, 
        help="depth of the view-dependence MLP"
    )
    snerg.add_argument(
        "--viewdir_netwidth", type=int, default=16,
        help="width of the view-dependence MLP"
    )
    snerg.add_argument(
        "--sparsity_strength", type=float, default=0.,
        help="lambda for sparsity loss"
    )
    snerg.add_argument(
        "--deg_view", type=int, default=4, 
        help="degree for positional encoding  of view direction"
    )
    snerg.add_argument(
        "--voxel_resolution", type=int, default=1000,
        help="resolution of voxel while baking"
    )

    return parser.parse_args()