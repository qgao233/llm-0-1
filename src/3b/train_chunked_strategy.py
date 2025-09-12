#!/usr/bin/env python3
"""
3B参数LLM模型分块训练策略脚本
实现新的数据训练策略：
- 每10个样本一块，每块训练5轮
- 20万样本需要100万轮训练
- 每1000轮保存checkpoint，只保留最新的
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import time
import json
import yaml
import math
import gc
import psutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from llmcore_3b import create_3b_model, get_device
from data_collectors import create_data_collectors, MixedDataset, DataProcessor

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_chunked_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChunkedDataset(torch.utils.data.Dataset):
    """分块数据集 - 支持按块训练"""
    
    def __init__(self, original_dataset, chunk_size=10):
        """
        初始化分块数据集
        
        Args:
            original_dataset: 原始数据集
            chunk_size: 每块的样本数量
        """
        self.original_dataset = original_dataset
        self.chunk_size = chunk_size
        self.total_samples = len(original_dataset)
        self.total_chunks = math.ceil(self.total_samples / chunk_size)
        
        logger.info(f"📦 分块数据集初始化:")
        logger.info(f"  📊 总样本数: {self.total_samples:,}")
        logger.info(f"  📦 块大小: {chunk_size}")
        logger.info(f"  🔢 总块数: {self.total_chunks:,}")
    
    def get_chunk(self, chunk_id):
        """获取指定块的数据"""
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        # 创建子集
        indices = list(range(start_idx, end_idx))
        chunk_dataset = Subset(self.original_dataset, indices)
        
        return chunk_dataset
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        return self.original_dataset[idx]


class ChunkedTrainingStrategy:
    """分块训练策略管理器"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = None
        self.is_multi_gpu = False
        self.is_save_interval = False
        
        # 训练策略参数（默认值）
        self.chunk_size = 10  # 每块样本数
        self.epochs_per_chunk = 5  # 每块训练轮数
        self.save_interval = 1000  # 每1000轮保存checkpoint
        self.current_global_step = 0  # 全局步数
        
        # 加载配置
        self._load_config()
        
        # 从配置文件更新训练策略参数
        self._update_strategy_params()
        
        # 设置设备
        self._setup_device()
        
        # 创建保存目录
        self._create_directories()
        
        # 初始内存清理
        self._cleanup_memory()
        
        logger.info("✅ 分块训练策略初始化完成")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✅ 配置文件加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _update_strategy_params(self):
        """从配置文件更新训练策略参数"""
        if not self.config:
            return
        
        chunked_config = self.config.get('chunked_training', {})
        
        # 更新参数
        self.chunk_size = chunked_config.get('chunk_size', self.chunk_size)
        self.epochs_per_chunk = chunked_config.get('epochs_per_chunk', self.epochs_per_chunk)
        self.save_interval = chunked_config.get('save_interval', self.save_interval)
        
        logger.info(f"🔧 分块训练策略参数:")
        logger.info(f"  📦 块大小: {self.chunk_size}")
        logger.info(f"  🔄 每块轮数: {self.epochs_per_chunk}")
        logger.info(f"  💾 保存间隔: {self.save_interval}")
    
    def _setup_device(self):
        """设置训练设备"""
        device_config = self.config.get('device', {})
        self.device, self.is_multi_gpu, available_gpus = get_device(
            force_cpu=device_config.get('force_cpu', False),
            device_config=device_config
        )
        
        # 设置混合精度
        if device_config.get('mixed_precision', True) and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("🔥 启用混合精度训练")
        
        logger.info(f"🎯 训练设备: {self.device}")
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['checkpoint']['save_dir'],
            self.config['monitoring']['log_dir'],
            'logs'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _prepare_data(self):
        """准备训练数据"""
        logger.info("📚 准备训练数据...")
        
        data_config = self.config['data']
        
        # 创建数据处理器
        tokenizer_type = "chinese_bpe" if data_config.get('use_bpe_tokenizer', True) else "char_level"
        vocab_path = data_config.get('vocab_cache_path', 'chinese_tokenizer_vocab_50k.json')
        
        self.data_processor = DataProcessor(
            tokenizer_type=tokenizer_type,
            vocab_path=vocab_path,
            vocab_size=self.config['model']['vocab_size'],
            config_path=self.config_path
        )
        
        # 创建数据收集器
        collectors, self.data_processor = create_data_collectors(self.config)
        
        if not collectors:
            raise ValueError("未能创建任何数据收集器")
        
        # 更新词汇表大小
        actual_vocab_size = self.data_processor.vocab_size
        if actual_vocab_size and actual_vocab_size != self.config['model']['vocab_size']:
            logger.info(f"🔄 动态调整词汇表大小: {self.config['model']['vocab_size']} -> {actual_vocab_size}")
            self.config['model']['vocab_size'] = actual_vocab_size
        
        # 创建混合数据集
        seq_length = self.config['model']['context_window']
        original_dataset = MixedDataset(
            collectors=collectors,
            seq_length=seq_length,
            config=data_config
        )
        
        # 创建分块数据集
        self.chunked_dataset = ChunkedDataset(original_dataset, self.chunk_size)
        
        logger.info(f"✅ 数据准备完成")
        logger.info(f"  📊 原始数据集大小: {len(original_dataset):,} 样本")
        logger.info(f"  📦 分块数据集: {self.chunked_dataset.total_chunks:,} 块")
        logger.info(f"  📏 序列长度: {seq_length}")
        
        return self.data_processor.itos if hasattr(self.data_processor, 'itos') else {}, \
               self.data_processor.stoi if hasattr(self.data_processor, 'stoi') else {}
    
    def _create_model(self):
        """创建3B模型"""
        logger.info("🏗️ 创建3B模型...")
        
        model_config = self.config['model']
        
        with torch.device(self.device):
            self.model = create_3b_model(model_config)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 启用梯度检查点以节省内存
        if model_config.get('use_gradient_checkpointing', True):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            logger.info("💾 启用梯度检查点")
        
        # 多GPU支持
        if self.is_multi_gpu:
            self.model = nn.DataParallel(self.model)
            logger.info("🚀 启用DataParallel多GPU训练")
        
        return self.model
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config['optimizer']
        
        lr = float(optimizer_config['lr'])
        weight_decay = float(optimizer_config['weight_decay'])
        eps = float(optimizer_config['eps'])
        betas = [float(b) for b in optimizer_config['betas']]
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        
        logger.info(f"🎯 优化器: AdamW, 学习率: {lr}")
    
    def _save_checkpoint(self, global_step, loss, is_best=False):
        """保存检查点 - 只保留最新的"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        
        # 准备检查点数据
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 删除旧的checkpoint
        old_checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        for old_checkpoint in old_checkpoints:
            try:
                old_checkpoint.unlink()
                logger.debug(f"🗑️ 删除旧检查点: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"⚠️ 删除旧检查点失败: {e}")
        
        # 保存新的checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💎 保存最佳模型: {best_path}")
        
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            logger.info(f"⚠️ 检查点不存在: {checkpoint_path}")
            return 0
        
        try:
            # 为了兼容PyTorch 2.6的安全机制，添加安全的全局变量
            import torch.serialization
            safe_globals = [
                'numpy.core.multiarray.scalar',
                'numpy.dtype',
                'numpy.ndarray',
                'collections.OrderedDict',
                'torch._utils._rebuild_tensor_v2',
                'torch._utils._rebuild_parameter',
                'torch.nn.parameter.Parameter',
                'builtins.dict',
                'builtins.list',
                'builtins.tuple',
                'builtins.int',
                'builtins.float',
                'builtins.str',
                'builtins.bool'
            ]
            
            # 使用安全的全局变量上下文管理器
            try:
                with torch.serialization.safe_globals(safe_globals):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except Exception as safe_load_error:
                logger.warning(f"⚠️ 安全加载失败，尝试传统加载方式: {safe_load_error}")
                # 回退到传统加载方式（不推荐，但为了兼容性）
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                logger.warning("⚠️ 使用了不安全的加载方式，请确保模型文件来源可信")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            global_step = checkpoint.get('global_step', 0)
            logger.info(f"✅ 检查点加载成功: {checkpoint_path}")
            logger.info(f"📊 恢复到全局步数: {global_step}")
            
            return global_step
            
        except Exception as e:
            logger.error(f"❌ 检查点加载失败: {e}")
            return 0
    
    def calculate_training_plan(self, total_samples):
        """
        计算训练计划 - 自动计算目标训练步数
        
        Args:
            total_samples: 总样本数
            
        Returns:
            训练计划字典
        """
        # 获取分块训练配置
        chunked_config = self.config.get('chunked_training', {})
        auto_calculate = chunked_config.get('auto_calculate_steps', True)
        target_multiplier = chunked_config.get('target_multiplier', 1000)
        
        total_chunks = math.ceil(total_samples / self.chunk_size)
        base_training_rounds = total_chunks * self.epochs_per_chunk
        
        if auto_calculate:
            # 自动计算：根据样本数量和倍数计算目标步数
            # 例如：20万样本 × 1000倍数 = 2亿步
            # 但这里我们按照您的逻辑：20万样本需要100万轮
            # 所以倍数应该是：100万 ÷ 20万 = 5
            # 更通用的公式：target_steps = total_samples * (target_multiplier / 200)
            # 这样当有20万样本时，target_multiplier=1000 会得到 20万 * (1000/200) = 100万步
            
            if total_samples >= 200000:  # 20万样本
                # 按照您的需求：20万样本需要100万轮
                target_total_steps = int(total_samples * (target_multiplier / 200))
            else:
                # 对于较少的样本，使用更合理的倍数
                # 确保至少每个样本被训练足够多次
                min_steps_per_sample = 5  # 每个样本至少训练5次
                calculated_steps = total_samples * min_steps_per_sample
                target_total_steps = max(calculated_steps, base_training_rounds * 2)
            
            logger.info(f"🧮 自动计算目标训练步数:")
            logger.info(f"  📊 样本数量: {total_samples:,}")
            logger.info(f"  🔢 目标倍数: {target_multiplier}")
            logger.info(f"  🎯 计算得出: {target_total_steps:,} 步")
        else:
            # 手动指定（向后兼容）
            target_total_steps = chunked_config.get('target_total_steps', 1000000)
            logger.info(f"📋 使用手动指定的目标步数: {target_total_steps:,}")
        
        # 计算需要重复多少次才能达到目标轮数
        repeat_cycles = math.ceil(target_total_steps / base_training_rounds)
        actual_total_steps = repeat_cycles * base_training_rounds
        
        plan = {
            'total_samples': total_samples,
            'chunk_size': self.chunk_size,
            'total_chunks': total_chunks,
            'epochs_per_chunk': self.epochs_per_chunk,
            'base_training_rounds': base_training_rounds,
            'target_total_steps': target_total_steps,
            'repeat_cycles': repeat_cycles,
            'actual_total_steps': actual_total_steps,
            'checkpoints_count': actual_total_steps // self.save_interval,
            'auto_calculated': auto_calculate
        }
        
        return plan
    
    def train_chunk(self, chunk_dataset, chunk_id, cycle_id):
        """训练单个数据块"""
        self.model.train()
        
        # 创建数据加载器
        batch_size = self.config['training']['batch_size']
        chunk_loader = DataLoader(
            chunk_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        chunk_losses = []
        
        # 训练指定轮数
        for epoch in range(self.epochs_per_chunk):
            epoch_loss = 0
            num_batches = len(chunk_loader)
            chunk_dataset = self.chunked_dataset.get_chunk(chunk_id)
            pbar = tqdm(chunk_loader, 
                       desc=f"周期{cycle_id+1} 块{chunk_id+1}(样本数: {len(chunk_dataset)})/{self.chunked_dataset.total_chunks} 轮{epoch+1}/{self.epochs_per_chunk}")
            
            for batch_idx, (x, y) in enumerate(pbar):
                try:
                    # 移动数据到设备
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    
                    # 前向传播
                    if self.scaler:
                        with autocast():
                            logits = self.model(x)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    else:
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # 立即删除logits以释放内存
                    del logits
                    
                    # 反向传播
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        
                        # 梯度裁剪
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # 统计
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    
                    # 更新全局步数
                    self.current_global_step += 1
                    
                    # 删除loss张量
                    del loss, x, y
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{loss_value:.4f}',
                        'global_step': self.current_global_step,
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # 保存检查点
                    if self.current_global_step % self.save_interval == 0:
                        self.is_save_interval = True
                        avg_loss = epoch_loss / (batch_idx + 1)
                        self._cleanup_memory()
                        self._save_checkpoint(self.current_global_step, avg_loss)
                        logger.info(f"📊 全局步数 {self.current_global_step:,}, 损失: {avg_loss:.4f}")
                    
                    # 定期内存清理
                    if batch_idx % 10 == 0:
                        self._cleanup_memory()
                
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
                        self._cleanup_memory()
                        raise
                    else:
                        raise
            
            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            chunk_losses.append(avg_epoch_loss)
            
        return np.mean(chunk_losses) if chunk_losses else 0.0
    
    def train(self):
        """开始分块训练"""
        logger.info("🚀 开始分块训练策略...")
        
        # 准备数据
        itos, stoi = self._prepare_data()
        
        # 计算训练计划
        plan = self.calculate_training_plan(len(self.chunked_dataset))
        
        logger.info("📋 训练计划:")
        logger.info(f"  📊 总样本数: {plan['total_samples']:,}")
        logger.info(f"  📦 总块数: {plan['total_chunks']:,}")
        logger.info(f"  🔄 每块训练轮数: {plan['epochs_per_chunk']}")
        logger.info(f"  📐 基础训练轮数: {plan['base_training_rounds']:,}")
        if plan['auto_calculated']:
            logger.info(f"  🎯 目标训练步数: {plan['target_total_steps']:,} (自动计算)")
        else:
            logger.info(f"  🎯 目标训练步数: {plan['target_total_steps']:,} (手动指定)")
        logger.info(f"  🔁 重复周期数: {plan['repeat_cycles']}")
        logger.info(f"  📈 实际总训练步数: {plan['actual_total_steps']:,}")
        logger.info(f"  💾 预计检查点数: {plan['checkpoints_count']:,}")
        
        # 创建模型和优化器
        self._create_model()
        self._create_optimizer()
        
        # 尝试加载最新的检查点
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        latest_checkpoint = None
        for checkpoint_file in checkpoint_dir.glob("checkpoint_step_*.pt"):
            if latest_checkpoint is None or checkpoint_file.stat().st_mtime > latest_checkpoint.stat().st_mtime:
                latest_checkpoint = checkpoint_file
        
        if latest_checkpoint:
            self.current_global_step = self._load_checkpoint(latest_checkpoint)
        
        # 训练循环
        best_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"🎯 开始训练，当前全局步数: {self.current_global_step}")
        
        for cycle in range(plan['repeat_cycles']):
            logger.info(f"\n🔄 开始训练周期 {cycle + 1}/{plan['repeat_cycles']}")
            
            cycle_losses = []
            
            # 遍历所有数据块
            for chunk_id in range(plan['total_chunks']):
                # 获取当前块的数据
                chunk_dataset = self.chunked_dataset.get_chunk(chunk_id)
                
                # 训练当前块
                chunk_loss = self.train_chunk(chunk_dataset, chunk_id, cycle)
                cycle_losses.append(chunk_loss)
                
                # 更新最佳损失
                if chunk_loss < best_loss and self.is_save_interval:
                    best_loss = chunk_loss
                    # 保存最佳模型
                    self._save_checkpoint(self.current_global_step, chunk_loss, is_best=True)
                
                # 内存清理
                self._cleanup_memory()
                self.is_save_interval = False
            
            # 周期结束统计
            avg_cycle_loss = np.mean(cycle_losses) if cycle_losses else 0.0
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ 周期 {cycle + 1} 完成")
            logger.info(f"  📊 平均损失: {avg_cycle_loss:.4f}")
            logger.info(f"  💎 最佳损失: {best_loss:.4f}")
            logger.info(f"  ⏱️ 已用时间: {elapsed_time/3600:.2f}小时")
            logger.info(f"  📈 全局步数: {self.current_global_step:,}")
            
            # 检查是否达到目标步数
            if self.current_global_step >= plan['actual_total_steps']:
                logger.info("🎯 已达到目标训练步数，训练完成！")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info("🎉 分块训练完成！")
        logger.info(f"💎 最佳损失: {best_loss:.4f}")
        logger.info(f"📈 总训练步数: {self.current_global_step:,}")
        logger.info(f"⏱️ 总训练时间: {total_time/3600:.2f}小时")
        
        # 最终保存
        self._save_checkpoint(self.current_global_step, best_loss, is_best=True)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='3B LLM分块训练策略脚本')
    parser.add_argument('--config', default='config/config_chunked_strategy.yaml', help='配置文件路径')
    parser.add_argument('--chunk-size', type=int, default=10, help='每块样本数量')
    parser.add_argument('--epochs-per-chunk', type=int, default=5, help='每块训练轮数')
    parser.add_argument('--save-interval', type=int, default=1000, help='保存间隔')
    
    args = parser.parse_args()
    
    print("🚀 启动分块训练策略...")
    print(f"📦 块大小: {args.chunk_size}")
    print(f"🔄 每块轮数: {args.epochs_per_chunk}")
    print(f"💾 保存间隔: {args.save_interval}")
    
    try:
        trainer = ChunkedTrainingStrategy(args.config)
        
        # 更新策略参数
        trainer.chunk_size = args.chunk_size
        trainer.epochs_per_chunk = args.epochs_per_chunk
        trainer.save_interval = args.save_interval
        
        trainer.train()
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
