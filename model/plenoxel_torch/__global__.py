BASIS_TYPE_SH = 1
BASIS_TYPE_3D_TEXTURE = 4
BASIS_TYPE_MLP = 255


def _get_c_extension():
    from warnings import warn

    try:
        import lib.plenoxel as _C

        if not hasattr(_C, "sample_grid"):
            _C = None
    except:
        _C = None

    if _C is None:
        warn(
            "CUDA extension svox2.csrc could not be loaded! "
            + "Operations will be slow.\n"
            + "Please do not import svox in the svox2 source directory."
        )
    return _C
