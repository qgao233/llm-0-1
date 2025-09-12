#!/usr/bin/env python3
"""
Checkpoint管理器模块
用于管理模型训练过程中的检查点保存、加载、清理等功能
支持预训练、微调等不同训练阶段的检查点管理
"""

import os
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import yaml
import json

logger = logging.getLogger(__name__)


class BaseCheckpointManager:
    """基础检查点管理器"""
    
    def __init__(self, save_dir: str, keep_last_n: int = 3, create_timestamp_dir: bool = True):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 检查点保存目录
            keep_last_n: 保留最近N个检查点
            create_timestamp_dir: 是否创建时间戳子目录
        """
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if create_timestamp_dir:
            # 创建时间戳目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = self.save_dir / f"checkpoint_{timestamp}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = self.save_dir
        
        logger.info(f"📁 Checkpoint目录: {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, step: int, loss: float, 
                       config: Dict, extra_info: Optional[Dict] = None, 
                       scaler: Optional[Any] = None, 
                       checkpoint_name: Optional[str] = None) -> Optional[Path]:
        """
        保存检查点
        
        Args:
            model: 模型实例
            optimizer: 优化器实例
            step: 训练步数
            loss: 当前损失
            config: 配置信息
            extra_info: 额外信息
            scaler: 混合精度训练的scaler
            checkpoint_name: 自定义检查点名称
            
        Returns:
            保存的检查点路径，失败时返回None
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 准备checkpoint数据
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 保存混合精度训练的scaler状态
        if scaler:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        if extra_info:
            checkpoint_data.update(extra_info)
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"💾 已保存checkpoint: {checkpoint_name} (loss: {loss:.4f})")
            
            # 清理旧的checkpoint
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
        except Exception as e:
            logger.error(f"❌ 保存checkpoint失败: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path], model, 
                       optimizer: Optional[Any] = None, 
                       scaler: Optional[Any] = None,
                       strict: bool = True) -> Optional[Dict]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            model: 模型实例
            optimizer: 优化器实例（可选）
            scaler: 混合精度训练的scaler（可选）
            strict: 是否严格匹配模型权重
            
        Returns:
            检查点数据字典，失败时返回None
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # 处理DataParallel包装的模型
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=strict)
            
            # 加载优化器状态
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载混合精度训练的scaler状态
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("🔥 已恢复混合精度训练状态")
            
            logger.info(f"✅ 已加载checkpoint: {checkpoint_path}")
            if 'step' in checkpoint:
                logger.info(f"  步数: {checkpoint['step']}")
            if 'loss' in checkpoint:
                logger.info(f"  损失: {checkpoint['loss']:.4f}")
            
            return checkpoint
        except Exception as e:
            logger.error(f"❌ 加载checkpoint失败: {e}")
            return None
    
    def save_best_model(self, model, optimizer, step: int, loss: float, 
                       config: Dict, extra_info: Optional[Dict] = None,
                       scaler: Optional[Any] = None) -> Optional[Path]:
        """保存最佳模型"""
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'is_best_model': True
        }
        
        if scaler:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        if extra_info:
            checkpoint_data.update(extra_info)
        
        try:
            torch.save(checkpoint_data, best_model_path)
            logger.info(f"💫 保存最佳模型 (loss: {loss:.4f})")
            return best_model_path
        except Exception as e:
            logger.error(f"❌ 保存最佳模型失败: {e}")
            return None
    
    def find_latest_checkpoint(self, pattern: str = "checkpoint_step_*.pt") -> Optional[Path]:
        """查找最新的检查点文件"""
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if not checkpoint_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📁 找到最新检查点: {latest_checkpoint}")
        return latest_checkpoint
    
    def list_checkpoints(self, pattern: str = "checkpoint_step_*.pt") -> List[Path]:
        """列出所有检查点文件"""
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files
    
    def _cleanup_old_checkpoints(self, pattern: str = "checkpoint_step_*.pt"):
        """清理旧的checkpoint文件"""
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if len(checkpoint_files) > self.keep_last_n:
            # 按修改时间排序
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            # 删除最旧的文件
            for old_file in checkpoint_files[:-self.keep_last_n]:
                try:
                    old_file.unlink()
                    logger.info(f"🗑️ 已清理旧checkpoint: {old_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理文件失败 {old_file.name}: {e}")
    
    def save_config(self, config: Dict, filename: str = "config.yaml"):
        """保存配置文件副本"""
        config_path = self.checkpoint_dir / filename
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"📝 配置文件已保存: {config_path}")
        except Exception as e:
            logger.error(f"❌ 保存配置文件失败: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Optional[Dict]:
        """获取检查点信息（不加载权重）"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            info = {
                'step': checkpoint.get('step', 'Unknown'),
                'loss': checkpoint.get('loss', 'Unknown'),
                'timestamp': checkpoint.get('timestamp', 'Unknown'),
                'file_size': Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
            }
            
            # 添加额外信息
            for key in ['epoch', 'shard', 'is_best_model']:
                if key in checkpoint:
                    info[key] = checkpoint[key]
            
            return info
        except Exception as e:
            logger.error(f"❌ 获取检查点信息失败: {e}")
            return None


class PretrainingCheckpointManager(BaseCheckpointManager):
    """预训练检查点管理器"""
    
    def __init__(self, save_dir: str, keep_last_n: int = 3):
        super().__init__(save_dir, keep_last_n, create_timestamp_dir=True)
        # 为预训练创建特殊的时间戳目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.save_dir / f"pretrain_{timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 预训练Checkpoint目录: {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, epoch: int, shard: int, step: int, 
                       loss: float, config: Dict, extra_info: Optional[Dict] = None, 
                       scaler: Optional[Any] = None) -> Optional[Path]:
        """
        保存预训练检查点
        
        Args:
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前轮数
            shard: 当前分片
            step: 训练步数
            loss: 当前损失
            config: 配置信息
            extra_info: 额外信息
            scaler: 混合精度训练的scaler
        """
        checkpoint_name = f"epoch_{epoch:02d}_shard_{shard:02d}_step_{step:06d}.pt"
        
        # 添加预训练特有的信息
        pretrain_info = {
            'epoch': epoch,
            'shard': shard,
            'training_type': 'pretraining'
        }
        
        if extra_info:
            pretrain_info.update(extra_info)
        
        return super().save_checkpoint(
            model, optimizer, step, loss, config, 
            extra_info=pretrain_info, scaler=scaler, 
            checkpoint_name=checkpoint_name
        )
    
    def save_best_model(self, model, optimizer, epoch: int, shard: int, step: int, 
                       loss: float, config: Dict, extra_info: Optional[Dict] = None,
                       scaler: Optional[Any] = None) -> Optional[Path]:
        """保存最佳预训练模型"""
        best_info = {
            'epoch': epoch,
            'shard': shard,
            'training_type': 'pretraining'
        }
        
        if extra_info:
            best_info.update(extra_info)
        
        return super().save_best_model(
            model, optimizer, step, loss, config, 
            extra_info=best_info, scaler=scaler
        )


class FineTuningCheckpointManager(BaseCheckpointManager):
    """微调检查点管理器"""
    
    def __init__(self, save_dir: str, keep_last_n: int = 2):
        super().__init__(save_dir, keep_last_n, create_timestamp_dir=True)
        # 为微调创建特殊的时间戳目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.save_dir / f"finetune_{timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 微调Checkpoint目录: {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, epoch: int, step: int, 
                       loss: float, config: Dict, extra_info: Optional[Dict] = None, 
                       scaler: Optional[Any] = None) -> Optional[Path]:
        """
        保存微调检查点
        
        Args:
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前轮数
            step: 训练步数
            loss: 当前损失
            config: 配置信息
            extra_info: 额外信息
            scaler: 混合精度训练的scaler
        """
        checkpoint_name = f"finetune_epoch_{epoch:02d}_step_{step:06d}.pt"
        
        # 添加微调特有的信息
        finetune_info = {
            'epoch': epoch,
            'training_type': 'finetuning'
        }
        
        if extra_info:
            finetune_info.update(extra_info)
        
        return super().save_checkpoint(
            model, optimizer, step, loss, config, 
            extra_info=finetune_info, scaler=scaler, 
            checkpoint_name=checkpoint_name
        )


class InferenceCheckpointManager(BaseCheckpointManager):
    """推理检查点管理器（主要用于加载）"""
    
    def __init__(self, model_dir: str):
        super().__init__(model_dir, keep_last_n=1, create_timestamp_dir=False)
    
    def find_best_model(self) -> Optional[Path]:
        """查找最佳模型文件"""
        best_model_path = self.checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info(f"💎 找到最佳模型: {best_model_path}")
            return best_model_path
        
        # 如果没有最佳模型，查找最新的检查点
        latest_checkpoint = self.find_latest_checkpoint("*.pt")
        if latest_checkpoint:
            logger.info(f"📁 使用最新检查点: {latest_checkpoint}")
            return latest_checkpoint
        
        logger.warning(f"⚠️ 在 {self.checkpoint_dir} 中未找到模型文件")
        return None


def create_checkpoint_manager(save_dir: str, manager_type: str = "base", 
                            **kwargs) -> BaseCheckpointManager:
    """
    检查点管理器工厂函数
    
    Args:
        save_dir: 保存目录
        manager_type: 管理器类型 ("base", "pretraining", "finetuning", "inference")
        **kwargs: 其他参数
    
    Returns:
        对应类型的检查点管理器
    """
    manager_type = manager_type.lower()
    
    if manager_type == "pretraining":
        return PretrainingCheckpointManager(save_dir, **kwargs)
    elif manager_type == "finetuning":
        return FineTuningCheckpointManager(save_dir, **kwargs)
    elif manager_type == "inference":
        return InferenceCheckpointManager(save_dir)
    else:
        return BaseCheckpointManager(save_dir, **kwargs)


def resume_training_from_checkpoint(checkpoint_path: Union[str, Path], 
                                  model, optimizer, scaler=None) -> Dict:
    """
    从检查点恢复训练的便捷函数
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例
        scaler: 混合精度训练的scaler
    
    Returns:
        包含恢复信息的字典
    """
    manager = BaseCheckpointManager("./", keep_last_n=1, create_timestamp_dir=False)
    checkpoint_data = manager.load_checkpoint(checkpoint_path, model, optimizer, scaler)
    
    if checkpoint_data:
        resume_info = {
            'epoch': checkpoint_data.get('epoch', 0),
            'shard': checkpoint_data.get('shard', 0),
            'step': checkpoint_data.get('step', 0),
            'loss': checkpoint_data.get('loss', float('inf')),
            'training_type': checkpoint_data.get('training_type', 'unknown')
        }
        
        logger.info(f"🔄 从checkpoint恢复训练:")
        for key, value in resume_info.items():
            logger.info(f"  {key}: {value}")
        
        return resume_info
    
    return {}


if __name__ == "__main__":
    # 测试检查点管理器
    import argparse
    
    parser = argparse.ArgumentParser(description="检查点管理器测试")
    parser.add_argument("--save-dir", type=str, default="./test_checkpoints", 
                       help="检查点保存目录")
    parser.add_argument("--type", type=str, default="base", 
                       choices=["base", "pretraining", "finetuning", "inference"],
                       help="管理器类型")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 创建测试管理器
        manager = create_checkpoint_manager(args.save_dir, args.type)
        
        # 测试基本功能
        print(f"✅ 检查点管理器测试成功")
        print(f"📁 检查点目录: {manager.checkpoint_dir}")
        
        # 列出现有检查点
        checkpoints = manager.list_checkpoints()
        print(f"📋 现有检查点数量: {len(checkpoints)}")
        
        if checkpoints:
            for cp in checkpoints[:3]:  # 显示前3个
                info = manager.get_checkpoint_info(cp)
                if info:
                    print(f"  - {cp.name}: step={info['step']}, loss={info['loss']:.4f}")
        
    except Exception as e:
        print(f"❌ 检查点管理器测试失败: {e}")
        raise
