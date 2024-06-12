import numpy as np
import random
import os

from tqdm import tqdm
from torch.utils import data
from utils.preprocess import load_motion
from os.path import join as pjoin
from utils.quaternion import *
from utils.utils import rigid_transform, process_motion_interhuman

class InterHuman(data.Dataset):
    """
    InterHuman dataset
    """

    def __init__(self, opt, num_samples=-1):
    
        # Configuration variables
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15
        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1
        self.motion_rep = opt.MOTION_REP
        self.cache = opt.CACHE
        self.extended = opt.EXTENDED
        
        # Data structures
        self.motion_dict = {}
        self.data_list = []
        data_list = []

        # Load paths from the given split
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        # Suffle paths
        random.shuffle(data_list)

        if num_samples > 0:
            data_list = data_list[:num_samples]
            print("Using only {} samples".format(num_samples))

        # Load data
        index = 0
        root = pjoin(opt.DATA_ROOT, "motions_processed/person1")
        for file in tqdm(os.listdir(root)):

            # Comment if you want to use the whole dataset
            if file.split(".")[0]+"\n" not in data_list:
                continue

            motion_name = file.split(".")[0]
            file_path_person1 = pjoin(root, file)
            file_path_person2 = pjoin(root.replace("person1", "person2"), file)
            text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")

            # Load interaction texts and make the swaps
            texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
            texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                            .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]

            # If using extended version, load individual desciptions of the motions
            if self.extended:
                text_path_individual1 = file_path_person1.replace("motions_processed", "annots_individual").replace("npy", "txt")
                text_path_individual2 = file_path_person2.replace("motions_processed", "annots_individual").replace("npy", "txt")
         
                if not os.path.exists(text_path_individual1):
                    continue
                else:
                    texts_individual1 = [item.replace("\n", "") for item in open(text_path_individual1, "r").readlines()]
                    texts_individual2 = [item.replace("\n", "") for item in open(text_path_individual2, "r").readlines()]
                    
                    # Make the swaps of the individual descriptions
                    texts_individual1_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts_individual2]
                    texts_individual2_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts_individual1]

            # Load motion and check if it is too short and cache it if needed
            if self.cache:
                motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)
                motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)
                if motion1 is None:
                    continue

            if self.cache:
                self.motion_dict[index] = [motion1, motion2]
                self.motion_dict[index+1] = [motion1_swap, motion2_swap]
            else:
                self.motion_dict[index] = [file_path_person1, file_path_person2]
                self.motion_dict[index + 1] = [file_path_person1, file_path_person2]

            # Fill data structures depending on the variable of the dataset used
            if self.extended:
                self.data_list.append({
                    "name": motion_name,
                    "motion_id": index,
                    "swap":False,
                    "texts":texts,
                    "texts_individual1":texts_individual1_swap,
                    "texts_individual2":texts_individual2_swap,
                })

                if opt.MODE == "train":
                    self.data_list.append({
                        "name": motion_name+"_swap",
                        "motion_id": index+1,
                        "swap": True,
                        "texts": texts_swap,
                        "texts_individual1":texts_individual1,
                        "texts_individual2":texts_individual2,
                    })
            else:
                self.data_list.append({
                    "name": motion_name,
                    "motion_id": index,
                    "swap":False,
                    "texts":texts
                })

                if opt.MODE == "train":
                    self.data_list.append({
                        "name": motion_name+"_swap",
                        "motion_id": index+1,
                        "swap": True,
                        "texts": texts_swap,
                    })

            index += 2

        print("Total Dataset Size: ", len(self.data_list))

    def __len__(self):
        """
        Get the length of the dataset
        """
        return len(self.data_list)

    def __getitem__(self, item):
        """
        Get an item from the dataset
            param item: Index of the item to get
        """

        # Get the data from the dataset
        idx = item % self.__len__()
        data = self.data_list[idx]
        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]

        # Select a random text from the list and if extended also select the individual descriptions
        text = random.choice(data["texts"]).strip()
        if self.extended:
            text_individual1 = random.choice(data["texts_individual1"]).strip()
            text_individual2 = random.choice(data["texts_individual2"]).strip()

        # Load the motion
        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2

        # Get motion lenght and select a random segment
        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]
        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        # Swap the motions randomly
        if np.random.rand() > 0.5:
            motion1, motion2 = motion2, motion1

        # Process the motion
        motion1, root_quat_init1, root_pos_init1 = process_motion_interhuman(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_interhuman(motion2, 0.001, 0, n_joints=22)

        # Rotate motion 2
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)
        
        gt_motion1 = motion1
        gt_motion2 = motion2

        # Check if the motion is too short and pad it
        gt_length = len(gt_motion1)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)

        # Return the data
        if self.extended:
            return name, text, gt_motion1, gt_motion2, gt_length, text_individual1, text_individual2
        else:
            return name, text, gt_motion1, gt_motion2, gt_length
