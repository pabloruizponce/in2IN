import torch
import torch.nn as nn
import numpy as np

class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) * out_uncond
        return cfg_out
    

class ClassifierFreeSampleModelMultiple(nn.Module):
    def __init__(self, model, cfg_scale, cfg_scale_interaction, cfg_scale_individuals):
        super().__init__()
        self.model = model
        self.s = cfg_scale
        self.s_interaction = cfg_scale_interaction
        self.s_individuals = cfg_scale_individuals

    def forward(self, x, timesteps, cond=None, mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x, x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps, timesteps, timesteps], dim=0)
        if cond is not None:

            cond_full = cond
            
            cond_interaction = torch.zeros_like(cond)
            cond_interaction[:,:768] = cond[:,:768]

            cond_individuals = torch.zeros_like(cond)
            cond_individuals[:,768:] = cond[:,768:]

            cond = torch.cat([cond_full, cond_interaction, cond_individuals, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask, mask, mask], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask)

        out_cond = out[:B]
        out_cond_interaction = out[B:B*2]
        out_cond_individuals = out[B*2:B*3]
        out_uncond = out[B*3:]

        cfg_out = (self.s * out_cond) + (self.s_interaction * out_cond_interaction) + (self.s_individuals * out_cond_individuals) + ((1-(self.s+self.s_interaction+self.s_individuals)) * out_uncond)
        return cfg_out


class ClassifierFreeSampleDualMDM(nn.Module):

    def __init__(self, m_individual, m_interaction, s_individual, s_interaction, s_composition_func, s_composition_value):
        super().__init__()
        self.m_individual = m_individual  
        self.m_interaction = m_interaction
        self.s_individual = s_individual
        self.s_interaction = s_interaction
        self.s_composition = self.weight(s_composition_func, s_composition_value)


    def weight(self, func, value):  
        print(f"Diffusion Weight Scheduler func: {func}, value: {value}")

        if func == "exp":
            return lambda x: np.exp(-value * (1000 - x))[0]
        elif func == "exp-inv":
            return lambda x: 1 - np.exp(-value * (1000 - x))[0]
        elif func == "lin":
            return lambda x: 1 - ((1000 - x) / 1000)[0]
        elif func == "const":
            return lambda x: value
        else:
            raise ValueError("Unknown function")

    def forward(self, x, timesteps, cond=None, mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)

        if cond is not None:
            cond_combined = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        if mask is not None:
            mask_combined = torch.cat([mask, mask], dim=0)
        else:
            mask_combined = None

        out_interaction = self.m_interaction(x_combined, timesteps_combined, cond=cond_combined, mask=mask_combined)
        out_individual = self.m_individual(x_combined, timesteps_combined, cond=cond_combined, mask=mask_combined)

        out_interaction_cond = out_interaction[:B]
        out_interaction_uncond = out_interaction[B:]

        out_individual_cond = out_individual[:B]
        out_individual_uncond = out_individual[B:]

        cfg_out_interaction = (out_interaction_uncond + self.s_interaction * (out_interaction_cond - out_interaction_uncond)) 
        cfg_out_individual = (out_individual_uncond + self.s_individual * (out_individual_cond - out_individual_uncond)) 

        w = self.s_composition(timesteps.cpu().numpy())
        cfg_out = cfg_out_interaction + w * (cfg_out_individual - cfg_out_interaction)
        return cfg_out
    


