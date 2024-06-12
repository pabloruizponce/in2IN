import sys
sys.path.append(sys.path[0] + r"/../")

import os
import time
import wandb
import torch
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule, DataModuleHML3D
from utils.configs import get_config
from os.path import join as pjoin
import argparse
from models.utils.utils import CosineWarmupScheduler
from utils.utils import print_current_loss
from models.in2in import in2IN


os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg, mode):
        super().__init__()

        self.cfg = cfg
        self.automatic_optimization = False
        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        # Can be ["individual", "iteraction", "dual"]
        self.mode = mode

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters(ignore=['model'])

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.mode == "individual":
            optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            return [optimizer]
        else:
            scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
            return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        if self.mode == "individual":
            name, text, motion1, motion_lens = batch_data
            motions = motion1.detach().float() 
        elif self.mode == "interaction":
            name, text, motion1, motion2, motion_lens, text_individual1, text_individual2 = batch_data
            motion1 = motion1.detach().float() 
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text

        if self.mode == "interaction":
            batch["text_individual1"] = text_individual1
            batch["text_individual2"] = text_individual2

        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()

        print("Model Iterations", self.it)
        print("Model Epochs", self.epoch)


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                # log metrics to wandb
                self.log(tag, mean_loss[tag], on_step=True, on_epoch=False, prog_bar=True)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])


    def on_train_epoch_end(self):
        if self.mode == "interaction":
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add arguments
    parser.add_argument('--train', type=str, required=True, help='Training Configuration file')
    parser.add_argument('--model' , type=str, required=True, help='Model Configuration file')
    parser.add_argument('--data', type=str, required=True, help='Data Configuration file')
    parser.add_argument('--mode', type=str, required=True, help='Model mode (individual or interaction)')
    parser.add_argument('--resume', type=str, required=False, help='Resume training from checkpoint')
    parser.add_argument('--device', type=list_of_ints, required=True, help='Device to run the training')

    # Parse the arguments
    args = parser.parse_args()

    model_cfg = get_config(args.model)
    train_cfg = get_config(args.train)

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='in2IN', name=train_cfg.GENERAL.EXP_NAME)

    if args.mode == "individual":
        data_cfg = get_config(args.data).humanml3d
        datamodule = DataModuleHML3D(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    elif args.mode == "interaction":
        data_cfg = get_config(args.data).interhuman
        datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)

    model = in2IN(model_cfg, args.mode)
    litmodel = LitTrainModel(model, train_cfg, args.mode)

    # Checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=-1)
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=args.device, accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision='16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    if args.resume:
        trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)