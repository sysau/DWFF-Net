import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
import os
import numpy as np


class HabitatSegModel(pl.LightningModule):
    def __init__(self, mode, config, arch, encoder_name, **kwargs):
        super().__init__()
        self.mode = mode
        self.save_hyperparameters(config)
        self.config = config
        self.num_classes = config['model']['num_classes']
        self.learning_rate = config['model']['learning_rate']
        self.arch = arch
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=3,
            classes=self.num_classes, **kwargs,
        )

        self.ignore_index = 0

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore background class
        self.loss_weight = config['model'].get('loss_weight', 0.5)  # Used to balance two losses

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

    def _compute_and_log_metrics(self, stage):
        """Calculate and log metrics, return per-class values for printing or saving"""
        iou_per_class = self._get_metric(stage, "iou").compute()  # shape: [C]
        f1_per_class = self._get_metric(stage, "f1").compute()    # shape: [C]
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

        if stage in ["test"]:
            for i in range(self.num_classes):
                if i == self.ignore_index:
                    continue
                self.log(f"{stage}_iou_{i}", iou_per_class[i], sync_dist=True)
                self.log(f"{stage}_f1_{i}", f1_per_class[i], sync_dist=True)
                self.log(f"{stage}_recall_{i}", recall_per_class[i], sync_dist=True)
                self.log(f"{stage}_precision_{i}", precision_per_class[i], sync_dist=True)

        # Reset metrics (prepare for next epoch)
        self._reset_metrics(stage)

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def _common_step(self, batch, batch_idx, stage):
        if len(batch) == 3:
            images, masks, image_ids = batch
            has_image_ids = True
        else:
            images, masks = batch
            has_image_ids = False
            image_ids = None
            
        logits = self(images)

        dice_loss = self.dice_loss(logits, masks)
        focal_loss = self.focal_loss(logits, masks)

        total_loss = self.loss_weight * dice_loss + (1 - self.loss_weight) * focal_loss

        self.log(f'{stage}_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}_focal_loss', focal_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = logits.argmax(dim=1)

        if stage == 'test':
            if has_image_ids:
                self._save_predictions(preds, image_ids)
            self._update_metrics(stage, preds, masks)
        else:
            iou_metric = getattr(self, f'{stage}_iou')
            iou_metric.update(preds, masks)
            iou_metric.compute()
            self.log(f'{stage}_iou', iou_metric, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def on_train_epoch_start(self):
        self.train_iou.reset()

    def on_validation_epoch_start(self):
        self.val_iou.reset()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics('test')

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=0.04,
            lr=self.learning_rate
        )

        if self.config['model'].get('lr_scheduler') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['trainer']['max_epochs']
            )
            return [optimizer], [scheduler]

    def _save_predictions(self, preds, image_ids):
        """Save prediction results as int8 type npy files, create different directories based on arch parameter"""
        # Create save directory based on arch parameter
        save_dir = f"test_predictions_{self.arch}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to int8 and save
        preds_int8 = preds.to(torch.int8)
        
        for i, image_id in enumerate(image_ids):
            # Get individual prediction result
            pred = preds_int8[i].cpu().numpy()
            
            # Save as npy file
            save_path = os.path.join(save_dir, f"{image_id}.npy")
            np.save(save_path, pred)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=0.04,
            lr=self.learning_rate
        )

        if self.config['model'].get('lr_scheduler') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['trainer']['max_epochs']
            )
            return [optimizer], [scheduler]

        return optimizer


if __name__ == "__main__":
    import yaml
    import os

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model = HabitatSegModel(config, arch='Unet', encoder_name='resnet50')
    model.forward(torch.randint(0, 255, (1, 3, 1248+16, 1248+16), dtype=torch.uint8))

    print(model)
