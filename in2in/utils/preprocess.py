import numpy as np
from utils.utils import *

FPS = 30

def load_motion(file_path, min_length, swap=False):
    """
    Load motions from the original dataset with all the information needed to the convert them to Interhuman format
        :param file_path: path to the motion file
        :param min_length: minimum length of the motion
        :param swap: swap the left and right side of the motion
    """
    try:
        motion = np.load(file_path).astype(np.float32)
    except:
        print("error: ", file_path)
        return None, None
    
    # Reshape motion
    motion1 = motion[:, :22 * 3]
    motion2 = motion[:, 62 * 3:62 * 3 + 21 * 6]
    motion = np.concatenate([motion1, motion2], axis=1)

    # If the motion is to short, return none.
    if motion.shape[0] < min_length:
        return None, None

    # Swap
    if swap:
        motion_swap = swap_left_right(motion, 22)
    else:
        motion_swap = None
    
    return motion, motion_swap

def load_motion_hml3d(pos_file_path, rot_file_path , min_length):
    """
    Load motions from hml3d dataset with all the information needed to the convert them to Interhuman format
        :param pos_file_path: path to the position file
        :param rot_file_path: path to the rotation file
    """

    # Try to extract motions from the original dataset
    try:
        pos_motion = np.load(pos_file_path).astype(np.float32)
        rot_motion = np.load(rot_file_path).astype(np.float32)
    except:
        print("error: ", pos_motion)
        return None, None
   
    # Conver postition from (LENGHT, JOINTS, 3) to (LENGHT, JOINTS*3)
    pos_motion = pos_motion[:,:22]
    pos_motion = pos_motion.reshape(pos_motion.shape[0], -1)[:-1,:]

    # Extract relative rotations from HumanML3D representation 
    rot_motion = rot_motion[:,4+(21*3)+(22*3):4+(21*3)+(22*3)+(21*6)].reshape(rot_motion.shape[0], -1)

    # Concatenate position and rotation
    motion = np.concatenate([pos_motion, rot_motion], axis=1)

    if motion.shape[0] < min_length:
        return None, None

    return motion, None