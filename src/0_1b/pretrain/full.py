#!/usr/bin/env python3
"""
0.1B模型全参预训练脚本
实现三层嵌套循环的预训练架构：
1. 第一层：轮数循环 (epochs)
2. 第二层：分片循环 (shards) - 用于定期checkpoint保存
3. 第三层：步数循环 (steps) - 真正的样本处理循环
"""

import os
import sys
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import warnings
import gc
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加父目录到路径，确保能导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本项目模块
from llmcore import create_model
from data_collectors import create_data_collectors, MixedDataset
from utils import get_device
from config_manager import PretrainingConfig
from checkpoint_manager import PretrainingCheckpointManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)


class ShardDataset(Dataset):
    """
    分片数据集类，用于配合DataLoader使用
    """
    
    def __init__(self, mixed_dataset, shard_size: int, seq_length: int):
        """
        初始化分片数据集
        
        Args:
            mixed_dataset: 混合数据集实例
            shard_size: 分片大小（样本数量）
            seq_length: 序列长度
        """
        self.mixed_dataset = mixed_dataset
        self.shard_size = shard_size
        self.seq_length = seq_length
        
    def __len__(self):
        return self.shard_size
    
    def __getitem__(self, idx):
        """获取一个训练样本"""
        # 从混合数据集随机获取样本
        collector = np.random.choice(self.mixed_dataset.collectors, p=self.mixed_dataset.weights)
        x, y = collector.get_sample(self.seq_length)
        return x, y

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_dir = self.log_dir / f"pretrain_{timestamp}"
        self.current_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.training_history = []
        
        logger.info(f"📊 日志目录: {self.current_log_dir}")
    
    def log_step(self, epoch, shard, step, loss, lr, elapsed_time, samples_per_sec=None):
        """记录训练步骤"""
        log_entry = {
            'epoch': epoch,
            'shard': shard,
            'step': step,
            'loss': loss,
            'learning_rate': lr,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if samples_per_sec:
            log_entry['samples_per_sec'] = samples_per_sec
        
        self.training_history.append(log_entry)
        
        # 输出到控制台
        if samples_per_sec:
            logger.info(f"📈 Epoch {epoch:02d}/{shard:02d} Step {step:06d} | "
                       f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                       f"Speed: {samples_per_sec:.1f} samples/s")
        else:
            logger.info(f"📈 Epoch {epoch:02d}/{shard:02d} Step {step:06d} | "
                       f"Loss: {loss:.4f} | LR: {lr:.2e}")
    
    def save_training_log(self):
        """保存训练日志"""
        log_file = self.current_log_dir / "training_log.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            logger.info(f"📝 训练日志已保存: {log_file}")
        except Exception as e:
            logger.error(f"❌ 保存训练日志失败: {e}")

def create_optimizer(model, config):
    """创建优化器"""
    training_config = config['training']
    
    # 确保所有参数都是正确的数值类型
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config.get('weight_decay', 0.01))
    beta1 = float(training_config.get('beta1', 0.9))
    beta2 = float(training_config.get('beta2', 0.95))
    eps = float(training_config.get('eps', 1e-8))
    
    if training_config.get('optimizer', 'adamw').lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps
        )
    else:
        # 默认使用Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    logger.info(f"🔧 创建优化器: {type(optimizer).__name__}")
    logger.info(f"  学习率: {lr}")
    logger.info(f"  权重衰减: {weight_decay}")
    return optimizer

def create_lr_scheduler(optimizer, config):
    """创建学习率调度器"""
    training_config = config['training']
    scheduler_type = training_config.get('lr_scheduler', 'cosine')
    
    # 确保数值参数类型正确
    max_steps = int(training_config['max_steps'])
    learning_rate = float(training_config['learning_rate'])
    min_lr_ratio = float(training_config.get('min_lr_ratio', 0.1))
    
    if scheduler_type == 'cosine':
        # 余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=learning_rate * min_lr_ratio
        )
    elif scheduler_type == 'linear':
        # 线性衰减调度器
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=max_steps
        )
    else:
        # 无调度器
        scheduler = None
    
    if scheduler:
        logger.info(f"📚 创建学习率调度器: {scheduler_type}")
        logger.info(f"  最大步数: {max_steps}")
        logger.info(f"  最小学习率比例: {min_lr_ratio}")
    else:
        logger.info("📚 不使用学习率调度器")
    
    return scheduler

@torch.no_grad()
def evaluate_model(model, dataset, config, device, num_eval_batches=10):
    """评估模型性能"""
    model.eval()
    
    total_loss = 0.0
    batch_size = config['training']['batch_size']
    seq_length = config['model']['context_window']
    
    for _ in range(num_eval_batches):
        try:
            # 获取评估batch
            batch_x, batch_y = dataset.collectors[0].get_batch(
                batch_size, 
                config['model']['context_window']
            )
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            
            total_loss += loss.item()
            
        except Exception as e:
            logger.warning(f"⚠️ 评估批次失败: {e}")
            continue
    
    model.train()
    avg_loss = total_loss / num_eval_batches if num_eval_batches > 0 else float('inf')
    return avg_loss

def train_model(config_path: str, resume_checkpoint: str = None):
    """主训练函数"""
    logger.info("🚀 开始0.1B模型预训练")
    
    # 1. 加载配置
    config_manager = PretrainingConfig(config_path)
    config = config_manager.config
    
    # 2. 设备检测
    device, use_multi_gpu, available_gpus = get_device(
        force_cpu=False,
        device_config=config.get('hardware', {})
    )
    
    logger.info(f"🎯 训练设备: {device}")
    if use_multi_gpu:
        logger.info(f"🚀 启用多GPU训练，使用 {len(available_gpus)} 个GPU")
    else:
        logger.info(f"📱 使用单GPU训练")
    
    # 3. 创建数据收集器和数据集
    logger.info("📚 准备数据集...")
    try:
        collectors, data_processor = create_data_collectors(config)
        
        # 创建混合数据集
        dataset = MixedDataset(
            collectors=collectors,
            seq_length=config['model']['context_window'],
            weights=None,  # 使用默认权重
            config=config.get('data', {})
        )
        
        logger.info(f"✅ 数据集准备完成，总大小: {len(dataset):,}")
        
    except Exception as e:
        logger.error(f"❌ 数据集准备失败: {e}")
        raise
    
    # 4. 创建模型
    logger.info("🏗️ 创建模型...")
    try:
        # 从分词器获取实际词汇表大小
        actual_vocab_size = data_processor.vocab_size
        if actual_vocab_size:
            config['model']['vocab_size'] = actual_vocab_size
            logger.info(f"📚 使用分词器计算的词汇表大小: {actual_vocab_size:,}")
        else:
            # 如果分词器没有vocab_size，抛出异常
            raise ValueError("分词器必须提供词汇表大小(vocab_size)")
            
        model = create_model(config['model'])
        model = model.to(device)
        
        # 启用梯度检查点以节省内存
        if config.get('hardware', {}).get('gradient_checkpointing', True):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            logger.info("💾 启用梯度检查点")
        
        # 多GPU支持
        if use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info("🚀 启用DataParallel多GPU训练")
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算各部分参数量
        vocab_size = config['model']['vocab_size']
        d_model = config['model']['d_model']
        n_layers = config['model']['n_layers']
        ffn_dim = config['model']['ffn_dim']
        
        embedding_params = vocab_size * d_model
        transformer_params = n_layers * (3 * d_model * d_model + 2 * d_model * ffn_dim)
        tie_weights = config['model'].get('tie_weights', True)
        output_params = 0 if tie_weights else vocab_size * d_model
        
        logger.info(f"✅ 模型创建完成")
        logger.info(f"📊 词汇表大小: {vocab_size:,}")
        logger.info(f"📊 模型参数分布:")
        logger.info(f"  - Embedding层: {embedding_params:,} 参数")
        logger.info(f"  - Transformer层: {transformer_params:,} 参数")
        logger.info(f"  - 输出层: {output_params:,} 参数 (tie_weights={tie_weights})")
        logger.info(f"📊 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info(f"📊 可训练参数: {trainable_params:,}")
        logger.info(f"📊 模型大小: {total_params * 4 / 1024**2:.1f} MB (FP32)")
        
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        raise
    
    # 5. 创建优化器、调度器和混合精度支持
    optimizer = create_optimizer(model, config)
    scheduler = create_lr_scheduler(optimizer, config)
    
    # 设置混合精度训练
    scaler = None
    if config.get('training', {}).get('use_amp', True) and device.type == 'cuda':
        scaler = GradScaler()
        logger.info("🔥 启用混合精度训练")
    
    # 6. 创建checkpoint管理器和训练监控器
    checkpoint_manager = PretrainingCheckpointManager(
        save_dir=config['checkpoint']['save_dir'],
        keep_last_n=config['checkpoint'].get('keep_last_n_checkpoints', 3)
    )
    
    training_monitor = TrainingMonitor(
        log_dir=config['logging']['log_dir']
    )
    
    # 7. 恢复训练（如果指定了checkpoint）
    start_epoch = 0
    start_shard = 0
    start_step = 0
    
    if resume_checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint(resume_checkpoint, model, optimizer, scaler)
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch']
            start_shard = checkpoint_data['shard']
            start_step = checkpoint_data['step']
            logger.info(f"🔄 从checkpoint恢复训练: epoch {start_epoch}, shard {start_shard}, step {start_step}")
    
    # 8. 计算训练步数
    logger.info("🧮 计算训练步数...")
    dataset_size = len(dataset)
    steps_per_shard, steps_per_epoch = config_manager.calculate_training_steps(dataset_size)
    
    # 9. 开始三层嵌套循环训练
    logger.info("🏃‍♂️ 开始预训练循环...")
    
    training_config = config['training']
    num_epochs = training_config['num_epochs']
    shards_per_epoch = training_config['shards_per_epoch']
    
    global_step = start_step
    best_loss = float('inf')
    
    try:
        # 第一层循环：轮数 (Epochs)
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"🔥 开始第 {epoch + 1}/{num_epochs} 轮训练")
            epoch_start_time = time.time()
            
            # 重置数据收集器
            dataset.reset_all_collectors()
            
            # 第二层循环：分片 (Shards) - 用于定期保存checkpoint
            for shard in range(start_shard if epoch == start_epoch else 0, shards_per_epoch):
                logger.info(f"🧩 开始第 {shard + 1}/{shards_per_epoch} 分片")
                shard_start_time = time.time()
                shard_total_loss = 0.0
                
                # 创建分片数据集
                shard_samples = training_config['samples_per_shard']
                shard_dataset = ShardDataset(
                    mixed_dataset=dataset,
                    shard_size=shard_samples,
                    seq_length=config['model']['context_window']
                )
                
                # 创建DataLoader
                num_workers = 0 if use_multi_gpu else config.get('hardware', {}).get('dataloader_num_workers', 0)
                shard_loader = DataLoader(
                    shard_dataset,
                    batch_size=training_config['batch_size'],
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available() and not use_multi_gpu,  # 多GPU时不使用pin_memory
                    drop_last=True  # 确保batch大小一致，对多GPU训练很重要
                )
                
                # 使用单个进度条显示训练进度
                desc = f"Epoch {epoch+1}/{num_epochs} | Shard {shard+1}/{shards_per_epoch} | Loss: N/A | LR: N/A | Step: {global_step}"
                pbar = tqdm(shard_loader, desc=desc)
                
                for batch_idx, (x, y) in enumerate(pbar):
                    step_start_time = time.time()
                    
                    try:
                        # 移动数据到设备
                        x = x.to(device)
                        y = y.to(device)
                        
                        # 前向传播
                        optimizer.zero_grad()
                        
                        if scaler:
                            # 混合精度训练
                            with autocast():
                                logits = model(x)
                                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                            
                            # 反向传播
                            scaler.scale(loss).backward()
                            
                            # 梯度裁剪
                            if training_config.get('gradient_clip_val', 0) > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 
                                    training_config['gradient_clip_val']
                                )
                            
                            # 优化器步进
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # 常规训练
                            logits = model(x)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                            
                            # 反向传播
                            loss.backward()
                            
                            # 梯度裁剪
                            if training_config.get('gradient_clip_val', 0) > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 
                                    training_config['gradient_clip_val']
                                )
                            
                            # 优化器步进
                            optimizer.step()
                        
                        # 立即删除logits以释放内存
                        del logits
                        
                        # 学习率调度
                        if scheduler:
                            scheduler.step()
                        
                        # 更新统计
                        current_loss = loss.item()
                        shard_total_loss += current_loss
                        
                        # 计算性能指标
                        step_elapsed = time.time() - step_start_time
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        # 更新进度条描述
                        samples_per_sec = training_config['batch_size'] / step_elapsed if step_elapsed > 0 else 0
                        desc = (f"Epoch {epoch+1}/{num_epochs} | Shard {shard+1}/{shards_per_epoch} | "
                               f"Loss: {current_loss:.4f} | LR: {current_lr:.2e} | "
                               f"Step: {global_step} | Speed: {samples_per_sec:.1f} samples/s")
                        pbar.set_description(desc)
                        
                        # 记录训练步骤
                        if global_step % training_config.get('log_every_n_steps', 100) == 0:
                            training_monitor.log_step(
                                epoch=epoch,
                                shard=shard,
                                step=global_step,
                                loss=current_loss,
                                lr=current_lr,
                                elapsed_time=step_elapsed,
                                samples_per_sec=samples_per_sec
                            )
                        
                        global_step += 1
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(f"❌ GPU内存不足: {e}")
                            # 紧急清理
                            if 'x' in locals():
                                del x
                            if 'y' in locals():
                                del y
                            if 'logits' in locals():
                                del logits
                            if 'loss' in locals():
                                del loss
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise
                        else:
                            logger.error(f"❌ 训练步骤失败 (epoch {epoch}, shard {shard}, step {global_step}): {e}")
                            raise
                    except Exception as e:
                        logger.error(f"❌ 训练步骤失败 (epoch {epoch}, shard {shard}, step {global_step}): {e}")
                        continue
                
                # 关闭进度条
                pbar.close()
                
                # 分片完成后的处理
                shard_elapsed = time.time() - shard_start_time
                avg_shard_loss = shard_total_loss / len(shard_loader) if len(shard_loader) > 0 else 0
                
                logger.info(f"✅ 分片 {shard + 1} 完成")
                logger.info(f"  平均损失: {avg_shard_loss:.4f}")
                logger.info(f"  耗时: {shard_elapsed:.2f}s")
                logger.info(f"  处理批次数: {len(shard_loader)}")
                
                # 保存分片级checkpoint
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    shard=shard,
                    step=global_step,
                    loss=avg_shard_loss,
                    config=config,
                    scaler=scaler,
                    extra_info={
                        'shard_elapsed_time': shard_elapsed,
                        'steps_completed': global_step,
                        'batches_processed': len(shard_loader)
                    }
                )
                
                # 更新最佳损失
                if avg_shard_loss < best_loss:
                    best_loss = avg_shard_loss
                    # 保存最佳模型
                    checkpoint_manager.save_best_model(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        shard=shard,
                        step=global_step,
                        loss=best_loss,
                        config=config,
                        scaler=scaler
                    )
                
                # 内存清理
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 重置分片起始位置
            start_shard = 0
            
            # 轮次完成后的处理
            epoch_elapsed = time.time() - epoch_start_time
            
            logger.info(f"🎯 第 {epoch + 1} 轮训练完成")
            logger.info(f"  总耗时: {epoch_elapsed / 3600:.2f} 小时")
            logger.info(f"  最佳损失: {best_loss:.4f}")
            
            # 每轮结束后进行评估
            if len(dataset) > 0:
                eval_loss = evaluate_model(model, dataset, config, device)
                logger.info(f"📊 评估损失: {eval_loss:.4f}")
                
                # 记录评估结果
                training_monitor.log_step(
                    epoch=epoch,
                    shard=-1,  # 表示评估步骤
                    step=global_step,
                    loss=eval_loss,
                    lr=optimizer.param_groups[0]['lr'],
                    elapsed_time=0
                )
    
    except KeyboardInterrupt:
        logger.info("⏹️ 用户中断训练")
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存最终模型和日志
        logger.info("💾 保存最终模型和日志...")
        
        # 保存最终checkpoint
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch if 'epoch' in locals() else 0,
            shard=shard if 'shard' in locals() else 0,
            step=global_step,
            loss=best_loss,
            config=config,
            scaler=scaler,
            extra_info={
                'training_completed': True,
                'best_loss': best_loss
            }
        )
        
        # 保存训练日志
        training_monitor.save_training_log()
        
        # 保存配置文件副本
        config_backup_path = checkpoint_manager.checkpoint_dir / "config.yaml"
        with open(config_backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"🎉 预训练完成！")
        logger.info(f"📁 模型保存在: {checkpoint_manager.checkpoint_dir}")
        logger.info(f"💫 最佳损失: {best_loss:.4f}")
        logger.info(f"🔢 总训练步数: {global_step}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="0.1B模型预训练脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="恢复训练的checkpoint路径"
    )
    
    args = parser.parse_args()
    
    # 获取配置文件的绝对路径
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 开始训练
    train_model(config_path, args.resume)

if __name__ == "__main__":
    main()
