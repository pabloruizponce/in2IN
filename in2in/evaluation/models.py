import torch
import torch.nn as nn  
import clip
from models.utils.utils import PositionalEncoding, set_requires_grad


from models import *

class MotionEncoder(nn.Module):
    """
    Motion encoder module for feature extractor evaluation model
    """
    def __init__(self, cfg):
        """ 
        Initialize the motion encoder module
            :param cfg: model configuration file
        """
        super().__init__()

        # Model parameters
        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        # Model architecture
        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))
        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        """
        Forward pass of the motion encoder module
            :param batch: input batch
            :return batch: updated batch
        """ 

        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape
        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)
        
        # Embedding
        x_emb = self.embed_motion(x)
        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        # Masking
        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        # Positional encoder and transformer
        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])
        batch["motion_emb"] = motion_emb

        return batch

class InterCLIP(nn.Module):
    """
    InterCLIP model for feature extractor evaluation
    It is based in clip model and MotionEncoder
    """
    def __init__(self, cfg):
        """
        Initialize the InterCLIP model
            :param cfg: model configuration file
        """
        super().__init__()

        # Model parameters
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.motion_encoder = MotionEncoder(cfg)
        self.latent_dim = self.latent_dim

        # CLIP model
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1]))
        set_requires_grad(self.token_embedding, False)

        # Additional text encoding layers
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=cfg.FF_SIZE,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 512)

        # Losses
        self.clip_training = "text_"
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        self.loss_ce = nn.CrossEntropyLoss()


    def generate_src_mask(self, T, length):
        """
        Generate source mask for transformer
            :param T: sequence length
            :param length: sequence length
            :return src_mask: source mask
        """
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        """
        Encode motion features
            :param batch: input batch
            :return batch: updated batch
        """
        batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(batch["motions"].device)
        batch.update(self.motion_encoder(batch))
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch

    def encode_text(self, batch):
        """
        Encode text features
            :param batch: input batch
            :return batch: updated batch
        """
        device = next(self.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        out = self.textTransEncoder(pe_tokens)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)

        # Normalize
        batch['text_emb'] = out
        batch["text_emb"] = batch["text_emb"] / batch["text_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch


    def compute_loss(self, batch):
        """
        Wrapper for calculating the loss of the model
            :param batch: input batch
        """

        losses = {}
        losses["total"] = 0

        # Encode text and motion
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        # Compute clip losses
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def compute_clip_losses(self, batch):
        """
        Computing losses from the motion encoder and the text encoder
            :param batch: input batch
            :return mixed_clip_loss: mixed clip loss
            :return clip_losses: clip losses
        """
        mixed_clip_loss = 0.
        clip_losses = {}

        for d in self.clip_training.split('_')[:1]:
            if d == 'image':
                features = self.clip_model.encode_image(batch['images']).float()  # preprocess is done in dataloader
            elif d == 'text':
                features = batch['text_emb']
            motion_features = batch['motion_emb']

            # Normalized features
            features_norm = features / features.norm(dim=-1, keepdim=True)
            motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logit_scale = self.latent_scale ** 2
            logits_per_motion = logit_scale * motion_features_norm @ features_norm.t()
            logits_per_d = logits_per_motion.t()

            batch_size = motion_features.shape[0]
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

            # Compute losses
            ce_from_motion_loss = self.loss_ce(logits_per_motion, ground_truth)
            ce_from_d_loss = self.loss_ce(logits_per_d, ground_truth)
            clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

            clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
            clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
            clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
            mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def forward(self, batch):
        """
        Forward pass of the InterCLIP model
            :param batch: input batch
            :return batch: updated batch
        """
        return self.compute_loss(batch)

    
