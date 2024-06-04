import torch
import numpy as np
from utils.paramUtil import FACE_JOINT_INDX, HML_KINEMATIC_CHAIN, HML_RAW_OFFSETS, L_IDX1, L_IDX2
from utils.quaternion import *
from scipy.ndimage import filters
class Skeleton(object):
    """
    Skeleton class to handle the skeleton data such as 
    offsets, kinematic tree, forward and inverse kinematics
    """
    def __init__(self, offset, kinematic_tree, device):
        """
        Initialize the Skeleton class
            :param offset: Offset of the skeleton that we are using
            :param kinematic_tree: Kinematic tree of the skeleton
            :param device: Device to run the skeleton
        """
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        # Get Forward Direction
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        # Get Root Rotation
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        # Inverse Kinematics
        # quat_params (batch_size, joints_num, 4)
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)
                R_loc = qmul_np(qinv_np(R), rot_u_v)
                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints


def uniform_skeleton(positions, target_skeleton_path="data/motions_processed/person1/1.npy"):
    """
    Create a uniform skeleton from the source skeleton to the target skeleton
        :param positions: Source Skeleton motion
        :param target_skeleton_path: Target Skeleton path of the motion
        :return: New joints of the source skeleton in the target skeleton format
    """

    # Target Skeleton
    n_raw_offsets = torch.from_numpy(HML_RAW_OFFSETS)
    kinematic_chain = HML_KINEMATIC_CHAIN
    example_data = np.load(target_skeleton_path)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    target_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    target_offset = target_skel.get_offsets_joints(example_data[0])

    # Source Skeleton
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)

    # Forward Kinematics
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints