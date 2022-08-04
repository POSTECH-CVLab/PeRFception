import numpy as np


def random_pose(poses, num_frames=50):

    rot_diff = np.einsum("ilk, jlm -> ijkm", poses[:, :3, :3], poses[:, :3, :3])
    rot_angle = (
        np.arccos(
            np.clip(
                (rot_diff[:, :, 0, 0] + rot_diff[:, :, 1, 1] + rot_diff[:, :, 2, 2] - 1)
                / 2,
                -1.0,
                1.0,
            )
        )
        / np.pi
        * 180
    )
    ignore_self = np.logical_not(np.eye(len(rot_diff), dtype=np.bool))

    trans_mask = (
        np.linalg.norm(poses[:, None, :3, 3] - poses[None, :, :3, 3], axis=-1) < 0.5
    )
    rot_idx = np.where(
        np.logical_and(trans_mask, np.logical_and(rot_angle < 40, ignore_self))
    )
    n_candidates = len(rot_idx[0])
    ret = np.zeros((num_frames, 4, 4))
    indices = np.random.choice(n_candidates, num_frames, replace=True)
    t = np.random.rand(num_frames)
    axis, angle = R_to_axis_angle(rot_diff[rot_idx[0][indices], rot_idx[1][indices]])
    angle = angle * t
    pose_rot = R_axis_angle(angle, axis)

    trans_t = (
        t[:, None] * poses[rot_idx[0][indices], :3, 3]
        + (1 - t)[:, None] * poses[rot_idx[1][indices], :3, 3]
    )
    ret[:, :3, :3] = np.einsum(
        "ijk, ikl -> ijl", poses[rot_idx[0][indices], :3, :3], pose_rot
    )
    ret[:, :3, 3] = trans_t
    ret[:, 3, 3] = 1.0

    return ret

def pose_interp(poses, factor):

    pose_list = []
    for i in range(len(poses)):
        pose_list.append(poses[i])
        
        if i == len(poses) - 1:
            factor = 4 * factor

        next_idx = (i+1) % len(poses)
        axis, angle = R_to_axis_angle((poses[next_idx, :3, :3] @ poses[i, :3, :3].T)[None])
        for j in range(factor-1):
            ret = np.eye(4)
            j_fact = (j + 1) / factor 
            angle_j = angle * j_fact
            pose_rot = R_axis_angle(angle_j, axis)
            ret[:3, :3] = pose_rot @ poses[i, :3, :3]
            trans_t = (
                (1 - j_fact) * poses[i, :3, 3]
                + (j_fact) * poses[next_idx, :3, 3]
            )
            ret[:3, 3] = trans_t
            pose_list.append(ret)

    return np.stack(pose_list)



def R_axis_angle(angle, axis):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    len_angle = len(angle)
    matrix = np.zeros((len_angle, 3, 3))

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    # Multiplications (to remove duplicate calculations).
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # Update the rotation matrix.
    matrix[:, 0, 0] = x * xC + ca
    matrix[:, 0, 1] = xyC - zs
    matrix[:, 0, 2] = zxC + ys
    matrix[:, 1, 0] = xyC + zs
    matrix[:, 1, 1] = y * yC + ca
    matrix[:, 1, 2] = yzC - xs
    matrix[:, 2, 0] = zxC - ys
    matrix[:, 2, 1] = yzC + xs
    matrix[:, 2, 2] = z * zC + ca

    return matrix


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    len_matrix = len(matrix)
    axis = np.zeros((len_matrix, 3))
    axis[:, 0] = matrix[:, 2, 1] - matrix[:, 1, 2]
    axis[:, 1] = matrix[:, 0, 2] - matrix[:, 2, 0]
    axis[:, 2] = matrix[:, 1, 0] - matrix[:, 0, 1]

    # Angle.
    r = np.hypot(axis[:, 0], np.hypot(axis[:, 1], axis[:, 2]))
    t = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    theta = np.arctan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r[:, None]

    # Return the data.
    return axis, theta
