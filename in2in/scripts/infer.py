import sys
sys.path.append(sys.path[0] + r"/../../")

from collections import OrderedDict
import copy
import os.path
import argparse
import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np

from lightning import LightningModule
from utils.utils import MotionNormalizer, MotionNormalizerHML3D
from utils.plot import plot_3d_motion
from utils.paramUtil import HML_KINEMATIC_CHAIN
from utils.configs import get_config
from models.dualmdm import load_DualMDM_model
from models.in2in import in2IN

class LitGenModel(LightningModule):
    def __init__(self, model, cfg, save_folder, mode):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False
        self.save_folder = save_folder

        # train model init
        self.model = model

        self.save_folder = os.path.join("results",save_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.mode = mode

        if self.mode == "individual":
            self.normalizer = MotionNormalizerHML3D()
        else:
            self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []

        if self.mode == "individual":
            joint = mp_data.reshape(-1,22,3)
            mp_joint.append(joint)
        else:
            for i, data in enumerate(mp_data):
                if i == 0:
                    joint = data[:,:22*3].reshape(-1,22,3)
                else:
                    joint = data[:,:22*3].reshape(-1,22,3)

                mp_joint.append(joint)

        plot_3d_motion(result_path, HML_KINEMATIC_CHAIN, mp_joint, title=caption, fps=30)


    def generate_one_sample(self, prompt_interaction, prompt_individual1, prompt_individual2, name):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt_interaction"] = prompt_interaction

        if self.mode != "individual":
            batch["prompt_individual1"] = prompt_individual1
            batch["prompt_individual2"] = prompt_individual2

        window_size = 210
        motion_output = self.generate_loop(batch, window_size)
        result_path = f"{self.save_folder}/{name}.mp4"

        if self.mode == "individual":
            self.plot_t2m(motion_output,
                          result_path,
                          batch["prompt_interaction"])
        else:
            self.plot_t2m([motion_output[0], motion_output[1]],
                        result_path,
                        batch["prompt_interaction"])

    def generate_loop(self, batch, window_size):
        prompt_interaction = batch["prompt_interaction"]

        if self.mode != "individual":
            prompt_individual1 = batch["prompt_individual1"]
            prompt_individual2 = batch["prompt_individual2"]

        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        batch["text"] = [prompt_interaction]
        if self.mode != "individual":
            batch["text_individual1"] = [prompt_individual1]
            batch["text_individual2"] = [prompt_individual2]

        batch = self.model.forward_test(batch)

        if self.mode == "individual":
            motion_output = batch["output"][0].reshape(-1, 262)
            motion_output = self.normalizer.backward(motion_output.cpu().detach().numpy())
            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            return joints3d

        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())
        
        sequences = [[], []]
        for j in range(2):
            motion_output = motion_output_both[:,j]
            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)

        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add optional arguments
    parser.add_argument('--model', type=str, required=True, help='Model Configuration file')
    parser.add_argument('--infer', type=str, required=True, help='Infer Configuration file')
    parser.add_argument('--mode', type=str, required=True, help='Mode of the inference (individual, interaction, dual)')
    parser.add_argument('--out', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model')

    parser.add_argument('--text_interaction', type=str, required=True, help='Interaction prompt')
    parser.add_argument('--text_individual1', type=str, required=False, help='Individual 1 prompt')
    parser.add_argument('--text_individual2', type=str, required=False, help='Individual 2 prompt')
    parser.add_argument('--name', type=str, required=True, help='Name of the output file')

    # Parse the arguments
    args = parser.parse_args()

    model_cfg = get_config(args.model)
    infer_cfg = get_config(args.infer)

    if args.mode == "dual":
        model = load_DualMDM_model(model_cfg)
    else:
        model = in2IN(model_cfg, args.mode)
        model.load_state_dict(torch.load(model_cfg.CHECKPOINT), strict=True) 

    litmodel = LitGenModel(model, infer_cfg, args.out, mode=args.mode).to(torch.device("cuda:"+ args.device))
    litmodel.generate_one_sample(args.text_interaction, args.text_individual1, args.text_individual2, args.name)

