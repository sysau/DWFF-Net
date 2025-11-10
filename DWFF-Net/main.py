import os

import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import HabitatDataModule
from model import DinoV3SegModel  # 补充模型导入

def seed_everything(seed=42):
    import torch, numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Habitat Segmentation Trainer")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "train_test"],
        help="Running mode: train(training only), test(testing only), train_test(train then test)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Model checkpoint path used in test mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path"
    )
    args = parser.parse_args()
    print(args)

    # Load configuration file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("config")
    print(config)

    # Initialize data module
    dm = HabitatDataModule(
        root_dir=config['data']['root_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )
    dm.setup()
    config['model']['num_classes'] = dm.num_classes  # Get number of classes from dataset
    print("data module setup done")

    # Initialize model
    model = DinoV3SegModel(
        mode=args.mode,
        config=config,
    )

    exp_dir = config['model']['decoder']['type']

    logger = TensorBoardLogger("logs", name=exp_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(os.path.join(config['trainer']['checkpoint_path'], exp_dir)),
        filename='{epoch}-{val_iou:.4f}',
        save_top_k=2,
        verbose=True,
        monitor='val_iou',
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_iou',
        patience=20,
        verbose=True,
        mode='max'
    )   
 
    # Configure trainer parameters
    trainer_config = config['trainer']
    # Compatible with CPU training
    if trainer_config['accelerator'] == 'cpu' or not torch.cuda.is_available():
        trainer_config['devices'] = 1
        trainer_config['accelerator'] = 'cpu'
        trainer_config['strategy'] = 'auto'
        trainer_config['precision'] = '32'
        print("Running in CPU mode")

    # Initialize trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=8,
        devices=trainer_config['devices'],
        accelerator=trainer_config['accelerator'],
        strategy=trainer_config['strategy'] if trainer_config['devices'] != 1 else 'auto',
        max_epochs=trainer_config['max_epochs'],
        precision=trainer_config['precision'],
        logger=logger,
        callbacks=[checkpoint_callback] if args.mode in ['train', 'train_test'] else [],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        num_sanity_val_steps=0
    )

    
    # Training mode
    if args.mode in ['train', 'train_test']:
        print(f"Starting training, mode: {args.mode}")
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)
        #trainer.fit(model, datamodule=dm)

    # Test mode
    if args.mode in ['test', 'train_test']:
        print(f"Starting testing, mode: {args.mode}")
        # Determine checkpoint to use for testing
        if args.mode == 'test':
            # Test-only mode must specify checkpoint
            if not args.ckpt:
                raise ValueError("Test mode must specify model checkpoint path via --ckpt")
            ckpt_path = args.ckpt
        else:
            # Test after training, use best checkpoint
            ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None

        if ckpt_path:
            print(f"Using checkpoint: {ckpt_path}")
        else:
            print("Warning: Best checkpoint not found, using current model weights for testing")

        # Test-specific trainer (ensure device configuration is correct)
        test_trainer = pl.Trainer(
            accelerator=trainer_config['accelerator'],
            devices=1,
            logger=logger,
            num_sanity_val_steps=0
        )
        test_trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
