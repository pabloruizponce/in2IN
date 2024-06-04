import os
import numpy as np
import random

from tqdm import tqdm
from utils.utils import *
from torch.utils import data
from utils.preprocess import *
from os.path import join as pjoin

class HumanML3D(data.Dataset):
    """
    HumanML3D dataset
    """
    def __init__(self, opt):

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

        # Data structures
        self.motion_dict = {}
        self.data_list = []
        data_list = []

        # Load paths from the given split
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        # Suffle paths
        random.shuffle(data_list)

        # Load data
        index = 0
        motion_path = pjoin(opt.DATA_ROOT, "interhuman/")
        for file in tqdm(os.listdir(motion_path)):

            # Comment if you want to use the whole dataset
            if file.split(".")[0]+"\n" not in data_list:
                continue

            motion_name = file.split(".")[0]
            motion_file_path = pjoin(motion_path, file)
            text_path = motion_file_path.replace("interhuman", "texts").replace("npy", "txt")

            # Load motion and text
            texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
            motion1 = np.load(motion_file_path).astype(np.float32)
            
            # Check if the motion is too short
            if motion1.shape[0] < self.min_length:
                continue

            # Cache the motion if needed
            if self.cache:
                self.motion_dict[index] = motion1
            else:
                self.motion_dict[index] = motion_file_path

            self.data_list.append({
                "name": motion_name,
                "motion_id": index,
                "swap":False,
                "texts":texts
            })

            index += 1

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

        # Select a random text from the list
        text = random.choice(data["texts"]).strip().split('#')[0]

        # Load the motion
        if self.cache:
            full_motion1 = self.motion_dict[motion_id]
        else:
            file_path1 = self.motion_dict[motion_id]
            full_motion1 = np.load(file_path1).astype(np.float32)
            
        # Get motion lenght and select a random segment 
        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
        else:
            idx = 0
            gt_length = min(length, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]

        # Check if the motion is too short and pad it
        gt_motion1 = motion1
        gt_length = len(gt_motion1)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)

        # Return the data
        return name, text, gt_motion1, gt_length

