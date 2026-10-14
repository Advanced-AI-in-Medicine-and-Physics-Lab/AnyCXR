import os
import torch
from tqdm import tqdm
import json
import pickle
import csv
import math
import numpy as np

from typing import TYPE_CHECKING, cast, List, Tuple, Dict, Any, Optional
from pathlib import Path

from torch.amp import autocast, GradScaler

# Ensure imports are correct based on your project structure
from cxas.dataset import get_5fold_dataloaders, get_model_channels, MultiChannelSegDataset # Removed AugmentationPlanner as it's not a class to import directly here
from cxas.augmentation import DeterministicAugmentor, AugmentationPlanner, AugmentationConfig
from cxas.models.UNet.unet_model import UNet # Assuming UNet is modified to accept conditional inputs and JTR flags
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts, LinearLR, SequentialLR, _LRScheduler



class CXAS_Trainer:
    def __init__(self,
                    base_dir: str,
                    fold_id: int = 0,
                    gpus: str = "0",
                    lr: float = 1e-4,
                    batch_size: int = 2,
                    output_dir: str = "./checkpoints",
                    num_epochs: int = 100,
                    resume_training: bool = False,
                    # NEW PARAMETERS FOR TRAINING CONFIGURATION
                    selected_views: Optional[List[str]] = None,
                    selected_classes: Optional[List[str]] = None,
                    angle_encoding_type: str = "sin_cos",
                    progressive_training_resolutions: Optional[List[int]] = None,
                    # MODIFIED: Replaced enable_jtr with jtr_start_epoch
                    jtr_start_epoch: int = math.inf, # JTR starts at this epoch. Set to math.inf to disable JTR entirely.
                    jtr_c_dist: float = 0.0,  # JTR distance loss weight (base value)
                    jtr_c_recon: float = 0.0,  # JTR reconstruction loss weight (base value)
                    further_training = False,
                    augmentation_configs: Optional[List[AugmentationConfig]] = None,
                    accumulation_steps = 1,
                    use_variations: bool = False,
                    ):
        self.device = torch.device(f"cuda:{gpus.split(',')[0]}" if torch.cuda.is_available() and gpus else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_epochs = num_epochs
        self.selected_views = selected_views
        self.selected_classes = selected_classes
        self.angle_encoding_type = angle_encoding_type
        self.progressive_training_resolutions = sorted(progressive_training_resolutions) if progressive_training_resolutions else [None]
        
        # Store JTR activation epoch and base weights
        self.jtr_start_epoch = jtr_start_epoch
        self.base_jtr_c_dist = jtr_c_dist
        self.base_jtr_c_recon = jtr_c_recon

        self.base_dir = base_dir
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.num_workers = 8
        self.use_variations = use_variations

        self.accumulation_steps = accumulation_steps

        self.log_file_path = os.path.join(self.output_dir, "training_log.csv")
        self.aug_planner = AugmentationPlanner(
            base_dir=base_dir,
            output_dir=output_dir,  # 和生成种子时的base_dir一致
            num_epochs=self.num_epochs,  # 和生成种子时的total_epochs一致（比如1000）
            augmentation_configs=augmentation_configs  # 和生成种子时的配置列表一致
        )

        print(f"Successfully initialized AugmentationPlanner with seeds from: {Path(output_dir)/'augmentation_seeds.pkl'}")

        dataset_json_path = os.path.join(base_dir, "dataset.json")
        in_channels, n_classes = get_model_channels(dataset_json_path, selected_classes=self.selected_classes)

        # Pass enable_jtr=True to UNet if JTR is ever going to be active (i.e., start epoch is not infinity)
        # This tells UNet to build its JTR_Encoder/Decoder components.
        unet_enable_jtr_build = (self.jtr_start_epoch != math.inf)

        self.model = UNet(
            model_name="resnet50",
            in_channels=in_channels,
            n_classes=n_classes,
            angle_encoding_type=self.angle_encoding_type,
            enable_jtr=unet_enable_jtr_build, # Pass this derived flag for UNet construction
        ).to(self.device)
        self.initial_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,  # Typically works well with AdamW
            betas=(0.9, 0.999)  # Default values same as Adam
        )
        self.clip_gradients = True
        self.epoch = 0
        self.scheduler = self._create_precomputed_lr_scheduler()
        self.best_loss = math.inf
        self.best_no_jtr_loss = math.inf
        self.best_jtr_loss = math.inf
        self.current_resolution_idx = 0
        self.current_resolution = None
        self.mean = 0.0
        self.std = 1.0
        if resume_training:
            latest_checkpoint_path = os.path.join(self.output_dir, "latest.pth")
            if os.path.exists(latest_checkpoint_path):
                print(f"Resuming training from {latest_checkpoint_path}")
                self.load_checkpoint(latest_checkpoint_path, further_training)
                if os.path.exists(self.log_file_path):
                    self._load_latest_log_entry()
                if further_training:
                    self.epoch = 0
            else:
                print("No checkpoint found for resume. Starting new training.")
                # 无checkpoint时，执行与新训练相同的逻辑
                self._compute_and_set_normalization_stats()  # 调用提取的方法
        else:
            # 新训练：直接计算标准化参数
            self._compute_and_set_normalization_stats()  # 调用提取的方法

        if not os.path.exists(self.log_file_path) or not resume_training:
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'best_val_loss_so_far', 'resolution', 'CE', 'Dice', 'jtr_dist', 'jtr_recon'])


    def _save_training_log_entry(self, epoch, train_loss, val_loss, learning_rate, best_val_loss_so_far, resolution, ce, dc, jtr_dist, jtr_recon):
        """Appends a new entry to the training log CSV with 3 decimal % format."""
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss * 100:.3f}%",
                f"{val_loss * 100:.3f}%",
                f"{learning_rate:.3e}",
                f"{best_val_loss_so_far * 100:.3f}%",
                str(resolution),
                f"{ce * 100:.3f}%",
                f"{dc * 100:.3f}%",
                f"{jtr_dist * 100:.3f}%",
                f"{jtr_recon * 100:.3f}%"
            ])
            
    def _parse_percentage(self, value: str) -> float:
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.rstrip('%')) / 100.0
                    return float(value)

    def _load_latest_log_entry(self):
        """Loads the last entry from the training log to resume epoch, best_loss, and resolution stage."""
        if not os.path.exists(self.log_file_path):
            return

        with open(self.log_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            last_row = None
            for row in reader:
                last_row = row

            if last_row and len(last_row) >= 6:
                try:
                    self.epoch = int(last_row[0])
                    self.best_loss = self._parse_percentage(last_row[4])
                    
                    res_str = last_row[5]
                    if res_str == "None" or self.progressive_training_resolutions[0] is None:
                        loaded_res_val = None
                    else:
                        if res_str.startswith('(') and res_str.endswith(')'):
                            loaded_res_val = tuple(map(int, res_str.strip('()').split(', ')))
                        else:
                            loaded_res_val = int(res_str)

                    if loaded_res_val is not None and self.progressive_training_resolutions and self.progressive_training_resolutions[0] is not None:
                        try:
                            if isinstance(self.progressive_training_resolutions[0], int):
                                self.current_resolution_idx = self.progressive_training_resolutions.index(loaded_res_val[0] if isinstance(loaded_res_val, tuple) else loaded_res_val)
                            else:
                                self.current_resolution_idx = self.progressive_training_resolutions.index(loaded_res_val)

                            print(f"Log found: Resuming training from after Epoch {self.epoch}, Best Loss: {self.best_loss:.4f}, Resolution: {loaded_res_val}")
                            self.epoch += 1
                        except ValueError:
                            print(f"Warning: Loaded resolution '{res_str}' not found in progressive_training_resolutions. Resetting to first stage.")
                            self.epoch = 0
                            self.best_loss = math.inf
                            self.current_resolution_idx = 0
                    else:
                        self.epoch += 1
                        self.current_resolution_idx = 0

                except ValueError as e:
                    print(f"Error parsing training log entry: {e}. Starting new training.")
                    self.epoch = 0
                    self.best_loss = math.inf
                    self.current_resolution_idx = 0
            else:
                print("Training log is empty or malformed. Starting new training.")
                self.epoch = 0
                self.best_loss = math.inf
                self.current_resolution_idx = 0

    def train_model(self):
        self.model.train()
        

        total_num_stages = len(self.progressive_training_resolutions)
        epochs_per_stage = self.num_epochs // total_num_stages if total_num_stages > 0 else self.num_epochs
        if total_num_stages > 0 and self.num_epochs % total_num_stages != 0:
            print(f"Warning: total_epochs ({self.num_epochs}) is not evenly divisible by number of stages ({total_num_stages}). Epochs per stage might be uneven.")

        start_epoch_in_current_stage = 0
        if self.epoch > 0 and self.current_resolution_idx < total_num_stages:
            start_epoch_in_current_stage = self.epoch % epochs_per_stage
            if self.epoch == (self.current_resolution_idx * epochs_per_stage):
                start_epoch_in_current_stage = 0


        for stage_idx in range(self.current_resolution_idx, total_num_stages):
            raw_res = self.progressive_training_resolutions[stage_idx]
            if isinstance(raw_res, int):
                self.current_resolution = (raw_res, raw_res)
            else:
                self.current_resolution = raw_res 

            print(f"\n--- Starting Training Stage {stage_idx+1}/{total_num_stages} at Resolution: {self.current_resolution} ---")

            self.train_loader, self.val_loader = get_5fold_dataloaders(
                base_dir=self.base_dir,
                fold_id=self.fold_id,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                augmentation_planner=self.aug_planner,
                selected_views=self.selected_views,
                selected_classes=self.selected_classes,
                current_resolution=self.current_resolution,
                mean=self.mean,           # 传递计算得到的均值
                std=self.std,              # 传递计算得到的标准差
                use_variations=self.use_variations  # 传递变体开关
            )

            for ep_in_stage in range(start_epoch_in_current_stage, epochs_per_stage):
                self.epoch = (stage_idx * epochs_per_stage) + ep_in_stage 
                
                original_train_dataset = cast(MultiChannelSegDataset, self.train_loader.dataset.dataset)
                original_train_dataset.epoch = self.epoch
                original_train_dataset.augmentor = DeterministicAugmentor(self.aug_planner, self.epoch)

                self.model.train()
                running_loss = 0.0
                running_ce_loss = 0.0
                running_dc_loss = 0.0
                running_jtr_dist_loss = 0.0
                running_jtr_recon_loss = 0.0
                
                if not self.train_loader:
                    print(f"Warning: Train loader is empty for resolution {self.current_resolution}. Skipping stage.")
                    break

                # Determine dynamic JTR weights for the current epoch/stage
                dynamic_jtr_c_dist = 0.0
                dynamic_jtr_c_recon = 0.0
                
                # JTR is active if self.jtr_start_epoch is finite AND current epoch is >= jtr_start_epoch
                if self.jtr_start_epoch != math.inf and self.epoch >= self.jtr_start_epoch:
                    dynamic_jtr_c_dist = self.base_jtr_c_dist
                    dynamic_jtr_c_recon = self.base_jtr_c_recon
                    print(f"Info: JTR is active with c_dist={dynamic_jtr_c_dist}, c_recon={dynamic_jtr_c_recon} (Epoch {self.epoch}).")
                elif self.jtr_start_epoch != math.inf: # JTR is configured but not yet active due to warm-up
                    print(f"Info: JTR is inactive for warm-up (Epoch {self.epoch}, warm-up ends at {self.jtr_start_epoch}).")
                # else: JTR is entirely disabled (jtr_start_epoch == math.inf)


                for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} (Res {self.current_resolution})")):
                    data = batch['data'].to(self.device)
                    target = batch['label'].to(self.device)
                    class_weights = batch['class_weights'].to(self.device)
                    view_angle = batch['view_angle'].to(self.device)
                    sample_global_weights = torch.tensor(batch["meta"]["sample_global_weight"]).to(self.device)
                    total_global_weight = sample_global_weights.sum()
                    sample_normalized_weights = sample_global_weights / total_global_weight


                    model_input = {
                        'data': data,
                        'label': target,
                        'class_weights': class_weights,
                        'view_angle': view_angle,
                        'jtr_c_dist_weight': dynamic_jtr_c_dist, # Pass dynamically
                        'jtr_c_recon_weight': dynamic_jtr_c_recon # Pass dynamically
                    }
                    output = self.model(model_input)

                    per_sample_loss = output['loss']

                    batch_loss = (per_sample_loss * sample_normalized_weights).sum()
                    batch_loss.backward()

                    loss = output['loss']
                    ce_loss_scaled = output['ce_loss']
                    dc_loss = output['dc_loss']
                    L_dist = output.get('L_dist', 0.0)
                    L_recon = output.get('L_recon', 0.0)

                    if (idx + 1) % self.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    running_loss += loss.item()
                    running_ce_loss += ce_loss_scaled.item()
                    running_dc_loss += dc_loss.item()
                    if L_dist is not None:
                        running_jtr_dist_loss += L_dist
                    if L_recon is not None:
                        running_jtr_recon_loss += L_recon

                    num_batches = len(self.train_loader)
                    report_points = [
                            int(num_batches / 3),
                            int(2 * num_batches / 3),
                        ]
                    if idx in report_points:
                        tqdm.write(f"[Epoch {self.epoch+1}/{self.num_epochs}][Res {self.current_resolution}] Batch {idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

                if (len(self.train_loader) % self.accumulation_steps) != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                avg_train_loss = running_loss / len(self.train_loader)
                avg_train_ce_loss = running_ce_loss / len(self.train_loader)
                avg_train_dc_loss = running_dc_loss / len(self.train_loader)
                avg_train_jtr_dist_loss = running_jtr_dist_loss / len(self.train_loader)
                avg_train_jtr_recon_loss = running_jtr_recon_loss / len(self.train_loader)

                print(f"\n--- Epoch {self.epoch+1} Summary (Resolution: {self.current_resolution}): ---")
                
                print(f"Avg Train Loss: {avg_train_loss:.4f}")

                val_loss = self._validate()
                print(f"Validation Loss: {val_loss:.4f}")

                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                self.best_loss = min(self.best_no_jtr_loss, self.best_jtr_loss)
                self._save_training_log_entry(self.epoch + 1, avg_train_loss, val_loss, current_lr, self.best_loss, self.current_resolution, avg_train_ce_loss, avg_train_dc_loss, avg_train_jtr_dist_loss, avg_train_jtr_recon_loss)

                self.save_checkpoint()
                print("Saved latest checkpoint")

                jtr_active = self.jtr_start_epoch != math.inf and self.epoch >= self.jtr_start_epoch
                if jtr_active:
                    if val_loss < self.best_jtr_loss:
                        self.best_jtr_loss = val_loss
                        self.best_loss = min(self.best_no_jtr_loss, self.best_jtr_loss)
                        self.save_checkpoint(best_type="jtr")
                        print(f"New JTR best model saved (loss: {self.best_jtr_loss:.4f})")
                else:
                    if val_loss < self.best_no_jtr_loss:
                        self.best_no_jtr_loss = val_loss
                        self.best_loss = min(self.best_no_jtr_loss, self.best_jtr_loss)
                        self.save_checkpoint(best_type="no_jtr")
                        print(f"New non-JTR best model saved (loss: {self.best_no_jtr_loss:.4f})")
            
            start_epoch_in_current_stage = 0 
            self.current_resolution_idx = stage_idx + 1 
            if stage_idx == total_num_stages - 1:
                break


    def _validate(self):
        self.model.eval() # Set model to evaluation mode
        val_loss = 0.0
        # Check if val_loader has data
        if not self.val_loader:
            print("Warning: Validation loader is empty. Skipping validation.")
            return math.inf # Return inf if no validation data

        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in tqdm(self.val_loader, desc="Validation"):
                data = batch['data'].to(self.device)
                target = batch['label'].to(self.device)
                class_weights = batch['class_weights'].to(self.device)
                view_angle = batch['view_angle'].to(self.device)

                model_input = {
                    'data': data,
                    'label': target,
                    'class_weights': class_weights,
                    'view_angle': view_angle,
                    'jtr_c_dist_weight': self.base_jtr_c_dist if self.jtr_start_epoch != math.inf and self.epoch >= self.jtr_start_epoch else 0.0,
                    'jtr_c_recon_weight': self.base_jtr_c_recon if self.jtr_start_epoch != math.inf and self.epoch >= self.jtr_start_epoch else 0.0,
                }
                output = self.model(model_input)
                val_loss += output['loss'].item()
        
        val_loss /= len(self.val_loader)
        self.model.train() # Set model back to train mode
        return val_loss

    def save_checkpoint(self, best_type=None):
        """Saves latest checkpoint and optional phase-specific best checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_loss": self.best_loss, 
            "best_no_jtr_loss": self.best_no_jtr_loss,
            "best_jtr_loss": self.best_jtr_loss,
            "current_resolution_idx": self.current_resolution_idx,
            # Save trainer config for full reproducibility
            "selected_views": self.selected_views,
            "selected_classes": self.selected_classes,
            "angle_encoding_type": self.angle_encoding_type,
            "progressive_training_resolutions": self.progressive_training_resolutions,
            # MODIFIED: Save jtr_start_epoch and base_jtr_c_dist/recon instead of enable_jtr
            "jtr_start_epoch": self.jtr_start_epoch,
            "base_jtr_c_dist": self.base_jtr_c_dist,
            "base_jtr_c_recon": self.base_jtr_c_recon,
            "jtr_latent_channels": getattr(self.model, 'jtr_latent_channels', None),  # Save JTR latent channels if available
            # Save LR scheduler state for robust resume
            "scheduler_state_dict": self.scheduler.state_dict(), # ADDED
            "lr": self.optimizer.param_groups[0]['lr'], # Still good to save current LR
            "num_epochs": self.num_epochs,
            # Also save model's in_channels and n_classes from model instance if available (useful for inference loading)
            "in_channels": getattr(self.model, 'in_channels', 1),
            "n_classes": getattr(self.model, 'n_classes', 0),
            "mean": torch.tensor(self.mean, dtype=torch.float32, device="cpu"),
            "std": torch.tensor(self.std, dtype=torch.float32, device="cpu"),
        }
        
        latest_path = os.path.join(self.output_dir, "latest.pth")
        best_no_jtr_path = os.path.join(self.output_dir, "best_no_jtr.pth")
        best_jtr_path = os.path.join(self.output_dir, "best_jtr.pth")
        
        torch.save(checkpoint, latest_path)
        
        if best_type == "no_jtr":
            torch.save(checkpoint, best_no_jtr_path)
        elif best_type == "jtr":
            torch.save(checkpoint, best_jtr_path)

    def load_checkpoint(self, path, further_training=False):
        """
        加载模型检查点：
        - further_training=True：仅加载模型权重，直接返回（不恢复其他状态）
        - further_training=False：完全恢复训练状态（模型、优化器、调度器、epoch等）
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # 始终加载模型权重

        # 恢复数据集的标准化参数
        self.mean = checkpoint.get("mean", self.mean).item()
        self.std = checkpoint.get("std", self.std).item()


        if further_training:
            print(f"Further training enabled. Only loaded model weights from checkpoint: {path}")
            return  # 直接返回，不恢复其他状态
        
        # 完全恢复训练状态（further_training=False）
        # 1. 恢复优化器状态
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 2. 恢复训练进度和关键指标
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", math.inf)
        self.best_no_jtr_loss = checkpoint.get("best_no_jtr_loss", self.best_loss)
        self.best_jtr_loss = checkpoint.get("best_jtr_loss", math.inf)
        self.current_resolution_idx = checkpoint.get("current_resolution_idx", 0)
        
        # 3. 恢复或创建调度器（使用检查点中的状态）
        self.scheduler = self._create_precomputed_lr_scheduler()  # 创建调度器
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 恢复调度器状态
            print("Restored LR scheduler state from checkpoint")
        else:
            # 若没有调度器状态，同步当前epoch（确保学习率索引正确）
            self.scheduler.last_epoch = self.epoch - 1
            print(f"No scheduler state in checkpoint. Initialized with epoch {self.epoch}")
        
        # 4. 恢复总轮数（使用当前设置的num_epochs）
        original_num_epochs = self.num_epochs
        print(f"Restored training from epoch {self.epoch}, total epochs set to {original_num_epochs}")
        self.num_epochs = original_num_epochs
        
        # 恢复其他训练配置参数
        #self.selected_views = checkpoint.get("selected_views", self.selected_views)
        #self.selected_classes = checkpoint.get("selected_classes", self.selected_classes)
        self.angle_encoding_type = checkpoint.get("angle_encoding_type", self.angle_encoding_type)
        self.progressive_training_resolutions = checkpoint.get("progressive_training_resolutions", self.progressive_training_resolutions)
        
        # 恢复JTR相关参数
        self.jtr_start_epoch = checkpoint.get("jtr_start_epoch", self.jtr_start_epoch)
        self.base_jtr_c_dist = checkpoint.get("base_jtr_c_dist", self.base_jtr_c_dist)
        self.base_jtr_c_recon = checkpoint.get("base_jtr_c_recon", self.base_jtr_c_recon)

        
    def predict(self, x: Dict[str, Any]): # Changed x to Dict for clarity on expected input structure
        """Inference interface - outputs sigmoid probabilities.
                Assumes input x is a dictionary including at least 'data' (image tensor).
                'view_angle' is required for conditional models; a default (0.0) is used if missing.
        """
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            # Ensure data is a dictionary
            if not isinstance(x, dict) or 'data' not in x:
                raise ValueError("Predict input must be a dictionary with at least a 'data' key (image tensor).")

            model_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            # Add dummy view_angle if not provided, for conditional model compatibility
            if 'view_angle' not in model_input:
                print("Warning: 'view_angle' not in prediction input. Assuming view_angle=0.0 (AP view).")
                # Ensure it's a tensor of appropriate shape (e.g., [1]) for batching
                model_input['view_angle'] = torch.tensor(0.0, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Add dummy values for 'label' and 'class_weights' if model's forward expects them but they're not needed for predict
            model_input.setdefault('label', None)
            model_input.setdefault('class_weights', None)

            # The model's forward pass should handle the dictionary input and return logits
            output = self.model(model_input)
            if 'logits' not in output:
                raise RuntimeError("Model's forward method did not return 'logits' key in its output dictionary.")
            return output['logits'].sigmoid()



    def _create_precomputed_lr_scheduler(
        self,
        warmup_epochs=10,      # 预热轮数
        T_0=50,               # 第一个周期的长度（轮数）
        T_mult=1.6,              # 周期长度倍增因子（如 2 表示下一个周期长度翻倍）
        decay_factor=0.5,      # 每个周期初始 LR 的衰减因子
        eta_min=5e-8           # 最小学习率
    ):
        """创建预计算学习率调度器，支持 T_mult 参数"""
        total_epochs = self.num_epochs
        initial_lr = self.initial_lr
        # 1. 预计算预热阶段的学习率
        warmup_lrs = []
        if warmup_epochs > 0:
            start_lr = initial_lr * 0.01
            step = (initial_lr - start_lr) / warmup_epochs
            for i in range(warmup_epochs):
                warmup_lrs.append(start_lr + i * step)

        # 2. 预计算周期衰减阶段的学习率
        cycle_lrs = []
        remaining_epochs = total_epochs - warmup_epochs
        current_cycle = 0      # 当前周期数（从 0 开始）
        current_initial_lr = initial_lr  # 当前周期的初始 LR
        current_cycle_length = T_0       # 当前周期的长度

        while remaining_epochs > 0:
            # 每个周期的实际长度（不超过剩余轮数）
            cycle_length = min(current_cycle_length, remaining_epochs)
            key_positions = {
                "Epoch 1": 0,  # 周期内第1轮的索引（0-based）
                "Epoch 50%": int(cycle_length * 0.5),  # 周期中间位置
                "Epoch 100%": cycle_length - 1  # 周期最后1轮
            }
            # 计算当前周期内每轮的学习率（余弦衰减）
            for i in range(cycle_length):
                progress = i / cycle_length  # 当前周期内的进度 (0~1)
                lr = eta_min + 0.5 * (current_initial_lr - eta_min) * (1 + math.cos(math.pi * progress))
                cycle_lrs.append(lr)

                for pos_name, pos_idx in key_positions.items():
                    if i == pos_idx:
                        print(f"Cycle {current_cycle + 1}(length: {cycle_length}) {pos_name} LR: {lr:.8e}")


            # 进入下一个周期：更新初始 LR、周期长度和剩余轮数
            current_initial_lr *= decay_factor
            current_cycle_length = int(current_cycle_length * T_mult)  # 周期长度按 T_mult 增长
            current_cycle += 1
            remaining_epochs -= cycle_length

        # 3. 合并预热和周期阶段的学习率列表
        all_lrs = warmup_lrs + cycle_lrs
        all_lrs = all_lrs[:total_epochs]  # 确保长度等于总轮数

        return PrecomputedLRScheduler(self.optimizer, all_lrs)
    

    def _compute_and_set_normalization_stats(self):
        """
        整合逻辑：创建临时数据集 + 计算归一化后、标准化前的mean和std + 设置到实例变量
        """
        # 1. 创建临时数据集
        print("初始化临时数据集以计算标准化参数...")

        target_res = (512, 512)
        print(f"注意：当前标准化参数计算使用固定分辨率 {target_res[0]}x{target_res[1]}")
        temp_dataset = MultiChannelSegDataset(
            base_dir=self.base_dir,
            is_train=False,
            selected_views=self.selected_views,
            selected_classes=self.selected_classes,
            target_resolution=target_res  
        )


        # 2. 临时禁用数据集的标准化（避免干扰统计量计算）
        original_mean, original_std = temp_dataset.mean, temp_dataset.std
        temp_dataset.mean, temp_dataset.std = 0.0, 1.0  # 临时关闭标准化

        # 3. 随机采样部分样本（平衡效率和准确性）
        sample_size = min(100, len(temp_dataset))
        indices = np.random.choice(len(temp_dataset), sample_size, replace=False)
        pixels_sum = 0.0
        pixels_sq_sum = 0.0
        total_pixels = 0

        # 4. 遍历样本计算像素总和与平方和
        print("计算训练集的标准化参数...")
        for idx in tqdm(indices, desc="计算标准化参数"):
            try:
                sample = temp_dataset[idx]  # 已归一化、增强，但未标准化（因步骤2）
                data = sample["data"]  # 形状: [C, H, W]，范围: [0, 1]
                
                # 累加统计量
                pixels_sum += data.sum().item()
                pixels_sq_sum += (data **2).sum().item()
                total_pixels += data.numel()
            except Exception as e:
                print(f"跳过样本 {idx}: {str(e)}")
                continue

        # 5. 恢复数据集原始标准化参数
        temp_dataset.mean, temp_dataset.std = original_mean, original_std

        # 6. 校验并计算最终mean和std
        if total_pixels == 0:
            raise ValueError("无法计算标准化参数：没有有效的像素数据")
        self.mean = pixels_sum / total_pixels
        self.std = np.sqrt((pixels_sq_sum / total_pixels) - (self.mean** 2))

        # 7. 输出结果
        print(f"标准化参数: mean={self.mean:.4f}, std={self.std:.4f}")


class PrecomputedLRScheduler(_LRScheduler):
    """预计算所有 epoch 的学习率，训练时直接读取"""
    def __init__(self, optimizer, lr_list, last_epoch=-1):
        self.lr_list = lr_list  # 预计算好的学习率列表
        super().__init__(optimizer, last_epoch)  # 显式传递 last_epoch

    def get_lr(self):
        # 按当前 epoch 索引直接返回预计算的学习率
        current_epoch = self.last_epoch
        if current_epoch >= len(self.lr_list):
            return [self.lr_list[-1]] * len(self.optimizer.param_groups)
        return [self.lr_list[current_epoch]] * len(self.optimizer.param_groups)

    def state_dict(self):
        """保存调度器状态（主要是当前轮次）"""
        return {
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """恢复调度器状态"""
        self.last_epoch = state_dict['last_epoch']
