import torch
from dataclasses import dataclass
import model.plenoxel_torch.utils as utils
from typing import Optional, Union, Tuple, List

from model.plenoxel_torch.__global__ import _get_c_extension

_C = _get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of cuvol, svox1, nvol

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False  # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0  # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0  # Noise to add to sigma
    # (for the BG model; only if randomize=True)

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque

        return opt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w.float()
        spec.fx = float(self.fx_val)
        spec.fy = float(self.fy_val)
        spec.cx = float(self.cx_val)
        spec.cy = float(self.cy_val)
        spec.width = int(self.width)
        spec.height = int(self.height)
        spec.ndc_coeffx = float(self.ndc_coeffs[0])
        spec.ndc_coeffy = float(self.ndc_coeffs[1])
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda
