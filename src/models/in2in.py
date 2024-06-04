import torch
import clip

from torch import nn
from models.nets import in2INDiffusion
from models.utils.utils import set_requires_grad

class in2IN(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        # Mode can be individual interaction or dual
        self.mode = mode

        # DECODER (Denoiser)
        self.decoder = in2INDiffusion(cfg, mode, sampling_strategy=cfg.STRATEGY)

        # TEXT ENCODER (Trainable) - 1 FOR EACH MODEL #
        # INTERACTION
        if self.mode == "interaction" or self.mode == "dual":
            clipTransEncoderLayer_interaction = nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                batch_first=True)
            
            self.clipTransEncoder_interaction = nn.TransformerEncoder(
                clipTransEncoderLayer_interaction,
                num_layers=2)
            
            self.clip_ln_interaction = nn.LayerNorm(768)

        # INDIVIDUAL
        if self.mode == "individual" or self.mode == "dual":
            clipTransEncoderLayer_individual = nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                batch_first=True)
            
            self.clipTransEncoder_individual = nn.TransformerEncoder(
                clipTransEncoderLayer_individual,
                num_layers=2)
            
            self.clip_ln_individual = nn.LayerNorm(768)

        # CLIP MODEL (No trainable) 
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

    def compute_loss(self, batch):
        if self.mode == "dual":
            batch = self.text_process(batch, mode="interaction", out_name="cond_interaction")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual1", out_name="cond_interaction_individual1")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual2", out_name="cond_interaction_individual2")
            batch = self.text_process(batch, mode="individual",text_name="text_individual1", out_name="cond_individual_individual1")
            batch = self.text_process(batch, mode="individual",text_name="text_individual2", out_name="cond_individual_individual2")
        elif self.mode == "interaction" :
            batch = self.text_process(batch, mode="interaction", out_name="cond_interaction")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual1", out_name="cond_interaction_individual1")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual2", out_name="cond_interaction_individual2")
        elif self.mode == "individual":
            batch = self.text_process(batch, mode="individual", out_name="cond_individual_individual1")
        
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        if self.mode == "dual":
            batch = self.text_process(batch, mode="interaction", out_name="cond_interaction")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual1", out_name="cond_interaction_individual1")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual2", out_name="cond_interaction_individual2")
            batch = self.text_process(batch, mode="individual",text_name="text_individual1", out_name="cond_individual_individual1")
            batch = self.text_process(batch, mode="individual",text_name="text_individual2", out_name="cond_individual_individual2")
        elif self.mode == "interaction" :
            batch = self.text_process(batch, mode="interaction", out_name="cond_interaction")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual1", out_name="cond_interaction_individual1")
            batch = self.text_process(batch, mode="interaction",text_name="text_individual2", out_name="cond_interaction_individual2")
        elif self.mode == "individual":
            batch = self.text_process(batch, mode="individual", out_name="cond_individual_individual1")
        
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch, mode, text_name="text", out_name="cond"):
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch[text_name]

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        if mode == "individual":
            out = self.clipTransEncoder_individual(clip_out)
            out = self.clip_ln_individual(out)
        elif mode == "interaction":
            out = self.clipTransEncoder_interaction(clip_out)
            out = self.clip_ln_interaction(out)
        else:
            raise ValueError("Mode not recognized")

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch[out_name] = cond

        return batch
