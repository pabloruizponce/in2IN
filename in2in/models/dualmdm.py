

from in2in.models.in2in import in2IN
import torch

def load_DualMDM_model(model_cfg):
    """
    Load the I2I model with the 2 checkpoints
        :param model_cfg: Model Configuration file
        :return: I2I model
    """
    model = in2IN(model_cfg, mode="dual")
    print("Model Created")
    ckpt = torch.load(model_cfg.CHECKPOINT_INTERACTION)
    ckpt_individual = torch.load(model_cfg.CHECKPOINT_INDIVIDUAL)
    ckpt.update(ckpt_individual)
    model.load_state_dict(ckpt, strict=True)

    return model
    