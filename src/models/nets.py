from models.utils.cfg_sampler import ClassifierFreeSampleDualMDM, ClassifierFreeSampleModel, ClassifierFreeSampleModelMultiple
import torch
import random
import torch.nn as nn


from models.utils.blocks import TransformerBlockDoubleCond
from models.utils.gaussian_diffusion import LossType, ModelMeanType, ModelVarType, MotionDiffusion, create_named_schedule_sampler, get_named_beta_schedule, space_timesteps
from models.utils.layers import FinalLayer
from models.utils.utils import PositionalEncoding, TimestepEmbedder, zero_module

class in2INDiffusion(nn.Module):
    # Mode can be individual interaction or dual
    def __init__(self, cfg, mode, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy
        self.mode = mode

        # Setting wieghts
        if self.mode == "dual":
            self.cfg_weight_individual = cfg.CFG_WEIGHT_INDIVIDUAL
            self.cfg_weight_interaction = cfg.CFG_WEIGHT_INTERACTION
            self.cfg_composition_weight_func = cfg.W_FUNC
            self.cfg_composition_weight_value = cfg.W_VALUE
        elif self.mode == "interaction":
            self.cfg_weight = cfg.CFG_WEIGHT
            self.cfg_weight_individual = cfg.CFG_WEIGHT_INDIVIDUAL
            self.cfg_weight_interaction = cfg.CFG_WEIGHT_INTERACTION
        elif self.mode == "individual":
            self.cfg_weight = cfg.CFG_WEIGHT

        # Creaning network
        if self.mode =="dual":
            self.net_interaction = in2INDenoiser(self.nfeats, 
                                                latent_dim=self.latent_dim, 
                                                ff_size=self.ff_size, 
                                                num_layers=self.num_layers,
                                                num_heads=self.num_heads, 
                                                dropout=self.dropout, 
                                                activation=self.activation, 
                                                mode="dual_interaction")
            
            self.net_individual = in2INDenoiser(self.nfeats, 
                                            latent_dim=self.latent_dim, 
                                            ff_size=self.ff_size, 
                                            num_layers=self.num_layers,
                                            num_heads=self.num_heads, 
                                            dropout=self.dropout, 
                                            activation=self.activation, 
                                            mode="dual_individual")
        elif self.mode == "interaction":
            self.net_interaction = in2INDenoiser(self.nfeats, 
                                                latent_dim=self.latent_dim, 
                                                ff_size=self.ff_size, 
                                                num_layers=self.num_layers,
                                                num_heads=self.num_heads, 
                                                dropout=self.dropout, 
                                                activation=self.activation, 
                                                mode="interaction")
        elif self.mode == "individual":
            self.net_individual = in2INDenoiser(self.nfeats, 
                                            latent_dim=self.latent_dim, 
                                            ff_size=self.ff_size, 
                                            num_layers=self.num_layers,
                                            num_heads=self.num_heads, 
                                            dropout=self.dropout, 
                                            activation=self.activation, 
                                            mode="individual")


        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)
        timestep_respacing=[self.diffusion_steps]


        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            mode=self.mode
        )
        
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):

        if self.mode == "interaction":
            cond_interaction = batch["cond_interaction"]
            cond_interaction_individual1 = batch["cond_interaction_individual1"]
            cond_interaction_individual2 = batch["cond_interaction_individual2"]
            cond = torch.cat([cond_interaction, cond_interaction_individual1, cond_interaction_individual2], dim=1)
        elif self.mode == "individual":
            cond_individual_individual1 = batch["cond_individual_individual1"]
            cond = torch.cat([cond_individual_individual1], dim=1)

        x_start = batch["motions"]
        B,T = batch["motions"].shape[:2]

        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)

        seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)

        if self.mode == "interaction":
            model = self.net_interaction
        elif self.mode == "individual":
            model = self.net_individual

        output = self.diffusion.training_losses(
            model=model,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs={"mask":seq_mask,
                          "cond":cond,
                          },
        )
        return output

    def forward(self, batch):
        
        if self.mode == "dual":
            cond_interaction = batch["cond_interaction"]
            cond_interaction_individual1 = batch["cond_interaction_individual1"]
            cond_interaction_individual2 = batch["cond_interaction_individual2"]
            cond_individual_individual1 = batch["cond_individual_individual1"]
            cond_individual_individual2 = batch["cond_individual_individual2"]
            cond = torch.cat([cond_interaction, cond_interaction_individual1, cond_interaction_individual2, cond_individual_individual1, cond_individual_individual2], dim=1)
        elif self.mode == "interaction":
            cond_interaction = batch["cond_interaction"]
            cond_interaction_individual1 = batch["cond_interaction_individual1"]
            cond_interaction_individual2 = batch["cond_interaction_individual2"]
            cond = torch.cat([cond_interaction, cond_interaction_individual1, cond_interaction_individual2], dim=1)
        elif self.mode == "individual":
            cond_individual_individual1 = batch["cond_individual_individual1"]
            cond = torch.cat([cond_individual_individual1], dim=1)
        
        B = cond.shape[0]
        T = batch["motion_lens"][0]

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            mode = self.mode
        )

        if self.mode == "dual":
            self.cfg_model = ClassifierFreeSampleDualMDM(self.net_individual, self.net_interaction, self.cfg_weight_individual, self.cfg_weight_interaction, self.cfg_composition_weight_func, self.cfg_composition_weight_value)
            output = self.diffusion_test.ddim_sample_loop(
                self.cfg_model,
                (B, T, self.nfeats*2),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "mask":None,
                    "cond":cond,
                },
                x_start=None)
        elif self.mode == "interaction":
            self.cfg_model = ClassifierFreeSampleModelMultiple(self.net_interaction, self.cfg_weight, self.cfg_weight_interaction, self.cfg_weight_individual)
            output = self.diffusion_test.ddim_sample_loop(
                self.cfg_model,
                (B, T, self.nfeats*2),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "mask":None,
                    "cond":cond,
                },
                x_start=None)
        elif self.mode == "individual":
            self.cfg_model = ClassifierFreeSampleModel(self.net_individual, self.cfg_weight)
            output = self.diffusion_test.ddim_sample_loop(
                self.cfg_model,
                (B, T, self.nfeats),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "mask":None,
                    "cond":cond,
                },
                x_start=None)


        return {"output":output}

class in2INDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 mode,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.mode = mode
        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(TransformerBlockDoubleCond(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size, mode=self.mode))

        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))


    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a = x[...,:self.input_feats]

        if self.mode != "individual":
            x_b = x[...,self.input_feats:]

        if mask is not None:
            mask = mask[...,0]

        if self.mode == "dual_interaction" or self.mode == "interaction":
            emb = self.embed_timestep(timesteps) + self.text_embed(cond[:,:768])
            emb_individual1 = self.embed_timestep(timesteps) + self.text_embed(cond[:,768:768*2])
            emb_individual2 = self.embed_timestep(timesteps) + self.text_embed(cond[:,768*2:768*3])
        elif self.mode == "dual_individual":
            emb_individual1 = self.embed_timestep(timesteps) + self.text_embed(cond[:,768*3:768*4])
            emb_individual2 = self.embed_timestep(timesteps) + self.text_embed(cond[:,768*4:])
        elif self.mode == "individual":
            emb_individual1 = self.embed_timestep(timesteps) + self.text_embed(cond[:,:768])
        else:
            raise ValueError("Mode not recognized")

        a_emb = self.motion_embed(x_a)
        h_a_prev = self.sequence_pos_encoder(a_emb)

        if self.mode != "individual":
            b_emb = self.motion_embed(x_b)
            h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            if self.mode == "interaction" or self.mode == "dual_interaction":
                h_a = block(h_a_prev, h_b_prev, emb_individual1, emb, key_padding_mask)
                h_b = block(h_b_prev, h_a_prev, emb_individual2, emb, key_padding_mask)
            elif self.mode == "dual_individual":
                h_a = block(h_a_prev, None, emb_individual1, None, key_padding_mask)
                h_b = block(h_b_prev, None, emb_individual2, None, key_padding_mask)
            elif self.mode == "individual":
                h_a = block(h_a_prev, None, emb_individual1, None, key_padding_mask)
            else:
                raise ValueError("Mode not recognized")

            h_a_prev = h_a

            if self.mode == "dual_interaction" or self.mode == "interaction":
                h_b_prev = h_b


        output_a = self.out(h_a)

        if self.mode == "individual":
            output = torch.cat([output_a], dim=-1)
        else:
            output_b = self.out(h_b)
            output = torch.cat([output_a, output_b], dim=-1)

        return output