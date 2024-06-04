import math
import time
import torch
import numpy as np

from utils.paramUtil import FACE_JOINT_INDX, FID_L, FID_R, FACE_JOINT_INDX, HML_KINEMATIC_CHAIN, HML_RAW_OFFSETS
from utils.quaternion import *
from utils.rotation_conversions import *
from utils.skeleton import Skeleton, uniform_skeleton

### MOTION NORMZALIZERS ###

class MotionNormalizer():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x
    
    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x

class MotionNormalizerHML3D():
    def __init__(self):
        mean = np.load("./data/HumanML3D/mean_interhuman.npy")
        std = np.load("./data/HumanML3D/std_interhuman.npy")
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x


class MotionNormalizerTorch():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()

    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x


class MotionNormalizerTorchHML3D():
    def __init__(self):
        mean = np.load("./data/HumanML3D/mean_interhuman.npy")
        std = np.load("./data/HumanML3D/std_interhuman.npy")
        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x
    

### MOTION PROCESSORS ###

TRANS_MATRIX = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def process_motion_interhuman(motion, feet_thre, prev_frames, n_joints, flip=True, skel=False):
    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    if skel:
        positions = uniform_skeleton(positions)

    if flip:
        positions = np.einsum("mn, tjn->tjm", TRANS_MATRIX, positions)

    # Put on Floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    # XZ at origin
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init_for_all, positions)

    # Get Foot Contacts
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
        feet_l_h = positions[:-1,FID_L,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
        feet_r_h = positions[:-1,FID_R,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Get Joint Rotation Representation
    rot_data = rotations

    # Get Joint Rotation Invariant Position Represention
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    # Generate final motion vector
    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]


def process_motion_hml3d(motion, feet_thre, skel=True):

    # Uniform Skeleton
    if skel:
        positions = uniform_skeleton(motion)
    else:
        positions = motion

    # Put on Floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)

    # New ground truth positions
    global_positions = positions.copy()

    # Get Foot Contacts
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Quaternion and Cartesian representation
    r_rot = None

    def get_rifke(positions):
        # Local pose
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        # All pose face Z+
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(HML_RAW_OFFSETS, HML_KINEMATIC_CHAIN, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)
        # Quaternion to continuous 6D (seq_len, 4)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        # Root Linear Velocity (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        # Root Angular Velocity (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    full_rot_data = cont_6d_params

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Generate final motion vector
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity, full_rot_data

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)
    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


### UTILS ###

def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)



def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rigid_transform(relative, data):
    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data