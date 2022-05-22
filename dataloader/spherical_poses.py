import numpy as np
import torch

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float32)


def spherical_pose(theta, phi, radius, torch_output=False):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32) @ c2w
    # c2w = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32) @ c2w
    
    return torch.from_numpy(c2w) if torch_output else c2w


def spherical_poses(cam_trans):
    return np.stack(
        [
            spherical_pose(angle, -30.0, 0.5) @ cam_trans 
            for angle in np.linspace(-180,180,40+1)[:-1]
        ], 0
    )