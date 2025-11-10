import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from transformers import AutoModel
from decoders import get_decoder
from losses import get_loss_function
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
import os
import numpy as np


class DinoV3SegModel(pl.LightningModule):
    def __init__(self, mode, config):
        super().__init__()
        self.mode = mode
        self.save_hyperparameters(config)
        self.config = config
        self.num_classes = config['model']['num_classes']
        self.patch_size = 16
        self.ignore_index = 0

        if self.mode == 'test':
            self._create_metrics(stage="test")
        else:
            # Evaluation metrics - Overall metrics
            self.train_iou = MulticlassJaccardIndex(
                num_classes=self.num_classes,
                ignore_index=0,
                average="macro",
                sync_on_compute=True
            )
            self.val_iou = MulticlassJaccardIndex(
                num_classes=self.num_classes,
                ignore_index=0,
                average="macro",
                sync_on_compute=True
            )
        # Load DINOv3 model
        print(f"Loading backbone: {config['model']['name']}")
        self.backbone = AutoModel.from_pretrained(config['model']['name'])
        self.backbone.config.output_hidden_states = True
        # Freeze backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Configure decoder
        decoder_config = self.config['model']['decoder']
        feature_dim = self.backbone.config.hidden_size
        if decoder_config['type'] in ['Simple', 'SimpleLight']:
            self.multi_level_layers = [-1]
            # DINOv3 ViT-L has an embedding dim of 1024
            in_channels_list = [feature_dim]
        else:
            self.multi_level_layers = decoder_config['params']['multi_level_layers']
            in_channels_list = [feature_dim] * len(self.multi_level_layers)
        #self.multi_level_layers = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17,  -6, -5, -4, -3, -2, -1]
        print("multi_level_layers len", len(self.multi_level_layers))
        self.decoder = get_decoder(self.config, in_channels_list, self.num_classes)

        # Loss function
        self.seg_loss_fn = get_loss_function(self.config['model']['loss_function'])


    def _create_metrics(self, stage):
        """Create per-class IoU and F1 for each stage"""
        for metric_name, MetricClass in [
            ("iou", MulticlassJaccardIndex),
            ("f1", MulticlassF1Score),
            ("recall", MulticlassRecall),
            ("precision", MulticlassPrecision)
        ]:
            metric = MetricClass(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                average=None,
                validate_args=False,
                sync_on_compute=True
            )
            setattr(self, f"{stage}_{metric_name}", metric)

    def _update_metrics(self, stage, preds, target):
        """Update metrics for the specified stage"""
        self._get_metric(stage, "iou").update(preds, target)
        self._get_metric(stage, "f1").update(preds, target)
        self._get_metric(stage, "recall").update(preds, target)
        self._get_metric(stage, "precision").update(preds, target)

    def _reset_metrics(self, stage):
        """Reset metrics for the specified stage"""
        self._get_metric(stage, "iou").reset()
        self._get_metric(stage, "f1").reset()
        self._get_metric(stage, "recall").reset()
        self._get_metric(stage, "precision").reset()

    def _get_metric(self, stage, name):
        return getattr(self, f"{stage}_{name}")

    def _compute_and_log_metrics(self, stage, prefix=""):
        """Calculate and log metrics, return per-class values for printing or saving"""
        iou_per_class = self._get_metric(stage, "iou").compute()  # shape: [C]
        f1_per_class = self._get_metric(stage, "f1").compute()  # shape: [C]
        recall_per_class = self._get_metric(stage, "recall").compute()  # shape: [C]
        precision_per_class = self._get_metric(stage, "precision").compute()  # shape: [C]

        # Filter ignore_index
        valid_mask = torch.arange(self.num_classes, device=iou_per_class.device) != self.ignore_index
        valid_iou = iou_per_class[valid_mask]
        valid_f1 = f1_per_class[valid_mask]
        valid_recall = recall_per_class[valid_mask]
        valid_precision = precision_per_class[valid_mask]

        # Calculate averages (iou, f1)
        miou = valid_iou.mean()
        mf1 = valid_f1.mean()

        mrecall = valid_recall.mean()  
        mprecision = valid_precision.mean()  

        # Log average metrics
        self.log(f"{stage}_iou", miou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_f1", mf1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_recall", mrecall, on_epoch=True, prog_bar=True, sync_dist=True)  
        self.log(f"{stage}_precision", mprecision, on_epoch=True, prog_bar=True, sync_dist=True)  

        # Optional: Log IoU/F1 for each class
        if stage in ["test"]:
            for i in range(self.num_classes):
                if i == self.ignore_index:
                    continue
                self.log(f"{stage}_iou_{i}", iou_per_class[i], sync_dist=True)
                self.log(f"{stage}_f1_{i}", f1_per_class[i], sync_dist=True)
                self.log(f"{stage}_recall_{i}", recall_per_class[i], sync_dist=True)
                self.log(f"{stage}_precision_{i}", precision_per_class[i], sync_dist=True)

    def forward(self, x):
        original_size = x.shape[2:]
        self.backbone.eval()
        with torch.no_grad():
            # 通过 backbone 获取多层特征
            outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            # [:, 5:, :] used to remove CLS token Register token
            features_list = [hidden_states[i][:, 5:, :] for i in self.multi_level_layers]

            h, w = original_size
            patch_h, patch_w = h // self.patch_size, w // self.patch_size
            decoder_input = [f.reshape(x.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2) for f in features_list]

        logits = self.decoder(decoder_input, original_size)

        return {
            "logits": logits,
        }

    def _common_step(self, batch, batch_idx, stage):
        if len(batch) == 3:
            images, masks, image_ids = batch
            has_image_ids = True
        else:
            images, masks = batch
            has_image_ids = False
            image_ids = None
            
        outputs = self(images)
        logits = outputs["logits"]
        fused_weights = self.decoder.fused_weights

        # 1. Calculate segmentation loss
        seg_loss = self.seg_loss_fn(logits, masks)
        total_loss = seg_loss

        w = fused_weights                       # Will be passed below
        entropy = -(w * (w+1e-8).log()).sum(1).mean()
        entropy_lambda = 0.01                 # Fixed at 0.01 for now, can grid search later
        total_loss = total_loss - entropy_lambda * entropy
        self.log(f'{stage}_entropy', entropy, on_step=False, on_epoch=True, prog_bar=True)

        # Log loss
        self.log(f'{stage}_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_segloss', seg_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = logits.argmax(dim=1)
        
        # Save prediction results during testing phase
        if stage == 'test':
            # Save predictions as int8 type npy files
            if has_image_ids:
                self._save_predictions(preds, image_ids)
            self._update_metrics(stage, preds, masks)
        else:
            iou_metric = getattr(self, f'{stage}_iou')
            iou_metric.update(preds, masks)
            iou_metric.compute()
            self.log(f'{stage}_iou', iou_metric, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def _save_predictions(self, preds, image_ids):
        """Save prediction results as int8 type npy files"""
        # Create save directory
        save_dir = "test_predictions"
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to int8 and save
        preds_int8 = preds.to(torch.int8)
        
        for i, image_id in enumerate(image_ids):
            # Get individual prediction result
            pred = preds_int8[i].cpu().numpy()
            
            # Save as npy file
            save_path = os.path.join(save_dir, f"{image_id}.npy")
            np.save(save_path, pred)

    def on_train_epoch_start(self):
        self.train_iou.reset()

    def on_validation_epoch_start(self):
        self.val_iou.reset()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics('test')

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.decoder.parameters(), weight_decay=0.04, lr=self.config['model']['learning_rate'])

        if self.config['model']['lr_scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['trainer']['max_epochs']
            )
            return [optimizer], [scheduler]

        return optimizer


if __name__ == "__main__":
    from dataset import HabitatDataModule
    import yaml
    import os

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dm = HabitatDataModule(
        root_dir=config['data']['root_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )
    dm.setup()
    config['model']['num_classes'] = dm.num_classes

    model = DinoV3SegModel(config)
    model.forward(torch.randn(1, 3, 1248, 1248))

    print(model)