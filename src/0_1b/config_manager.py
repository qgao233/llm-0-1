#!/usr/bin/env python3
"""
配置管理器模块
用于管理0.1B模型的各种配置，包括预训练、微调、推理等配置
提供配置验证、默认值设置、训练步数计算等功能
"""

import os
import yaml
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseConfig:
    """基础配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def save_config(self, save_path: Optional[str] = None):
        """保存配置文件"""
        save_path = save_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"✅ 配置文件已保存: {save_path}")
        except Exception as e:
            logger.error(f"❌ 配置文件保存失败: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class PretrainingConfig(BaseConfig):
    """预训练配置管理器"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.validate_config()
        
    def validate_config(self):
        """验证配置文件的完整性"""
        required_sections = ['model', 'training', 'data', 'checkpoint', 'hardware']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        
        # 验证并设置训练配置默认值
        self._validate_training_config()
        
        # 验证并设置模型配置默认值
        self._validate_model_config()
        
        # 验证并设置数据配置默认值
        self._validate_data_config()
        
        # 验证并设置硬件配置默认值
        self._validate_hardware_config()
        
        # 打印配置摘要
        self._print_config_summary()
        
    def _validate_training_config(self):
        """验证并设置训练配置默认值"""
        training_config = self.config['training']
        
        # 必需的训练配置键
        required_keys = ['batch_size', 'learning_rate']
        for key in required_keys:
            if key not in training_config:
                raise ValueError(f"训练配置缺少必需的键: {key}")
        
        # 设置默认值
        defaults = {
            'num_epochs': 3,
            'shards_per_epoch': 5,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'max_steps': 50000,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-8,
            'lr_scheduler': 'cosine',
            'min_lr_ratio': 0.1,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'use_amp': True,
            'log_every_n_steps': 100
        }
        
        for key, default_value in defaults.items():
            if key not in training_config:
                training_config[key] = default_value
                logger.info(f"📝 设置训练配置默认值: {key} = {default_value}")
    
    def _validate_model_config(self):
        """验证并设置模型配置默认值"""
        model_config = self.config['model']
        
        # 必需的模型配置键
        required_keys = ['d_model', 'n_layers', 'n_heads', 'context_window']
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"模型配置缺少必需的键: {key}")
        
        # 设置默认值
        defaults = {
            'ffn_dim': model_config['d_model'] * 4,
            'dropout': 0.1,
            'tie_weights': True,
            'max_position_embeddings': model_config['context_window']
        }
        
        for key, default_value in defaults.items():
            if key not in model_config:
                model_config[key] = default_value
                logger.info(f"📝 设置模型配置默认值: {key} = {default_value}")
    
    def _validate_data_config(self):
        """验证并设置数据配置默认值"""
        data_config = self.config['data']
        
        # 设置默认值
        defaults = {
            'max_seq_length': self.config['model']['context_window'],
            'pad_token_id': 0,
            'eos_token_id': 2,
            'use_bpe_tokenizer': True,
            'vocab_cache_path': 'chinese_tokenizer_vocab_25k.json'
        }
        
        for key, default_value in defaults.items():
            if key not in data_config:
                data_config[key] = default_value
                logger.info(f"📝 设置数据配置默认值: {key} = {default_value}")
    
    def _validate_hardware_config(self):
        """验证并设置硬件配置默认值"""
        hardware_config = self.config['hardware']
        
        # 设置默认值
        defaults = {
            'device': 'auto',
            'multi_gpu': False,
            'gradient_checkpointing': True,
            'dataloader_num_workers': 4,
            'pin_memory': True
        }
        
        for key, default_value in defaults.items():
            if key not in hardware_config:
                hardware_config[key] = default_value
                logger.info(f"📝 设置硬件配置默认值: {key} = {default_value}")
    
    def _print_config_summary(self):
        """打印配置摘要"""
        training_config = self.config['training']
        model_config = self.config['model']
        
        logger.info(f"📊 配置摘要:")
        logger.info(f"  模型配置:")
        logger.info(f"    - d_model: {model_config['d_model']}")
        logger.info(f"    - n_layers: {model_config['n_layers']}")
        logger.info(f"    - n_heads: {model_config['n_heads']}")
        logger.info(f"    - context_window: {model_config['context_window']}")
        logger.info(f"    - ffn_dim: {model_config['ffn_dim']}")
        
        logger.info(f"  训练配置:")
        logger.info(f"    - 总轮数: {training_config['num_epochs']}")
        logger.info(f"    - 每轮分片数: {training_config['shards_per_epoch']}")
        logger.info(f"    - 批次大小: {training_config['batch_size']}")
        logger.info(f"    - 学习率: {training_config['learning_rate']}")
        logger.info(f"    - 优化器: {training_config['optimizer']}")
        
    def calculate_training_steps(self, dataset_size: int) -> Tuple[int, int]:
        """根据数据集大小计算训练步数"""
        training_config = self.config['training']
        batch_size = training_config['batch_size']
        shards_per_epoch = training_config['shards_per_epoch']
        
        # 计算每个分片的样本数（数据集总大小除以分片数）
        samples_per_shard = dataset_size // shards_per_epoch
        
        # 计算每个分片的步数（分片样本数除以批次大小）
        steps_per_shard = max(1, samples_per_shard // batch_size)
        
        # 计算每轮的步数（每分片步数乘以分片数）
        steps_per_epoch = steps_per_shard * shards_per_epoch
        
        # 更新配置
        training_config['steps_per_shard'] = steps_per_shard
        training_config['steps_per_epoch'] = steps_per_epoch
        training_config['samples_per_shard'] = samples_per_shard
        
        # 更新max_steps为总训练步数
        total_steps = steps_per_epoch * training_config['num_epochs']
        training_config['max_steps'] = total_steps
        
        logger.info(f"📊 训练步数计算结果:")
        logger.info(f"  数据集大小: {dataset_size:,}")
        logger.info(f"  每分片样本数: {samples_per_shard:,}")
        logger.info(f"  每分片步数: {steps_per_shard:,}")
        logger.info(f"  每轮步数: {steps_per_epoch:,}")
        logger.info(f"  总训练步数: {total_steps:,}")
        
        return steps_per_shard, steps_per_epoch
    
    def calculate_model_parameters(self) -> Dict[str, int]:
        """计算模型参数量"""
        model_config = self.config['model']
        
        # 从配置中获取参数，vocab_size可能动态确定
        vocab_size = model_config.get('vocab_size', 25000)  # 默认25k
        d_model = model_config['d_model']
        n_layers = model_config['n_layers']
        n_heads = model_config['n_heads']
        ffn_dim = model_config['ffn_dim']
        tie_weights = model_config['tie_weights']
        
        # 计算各部分参数量
        embedding_params = vocab_size * d_model
        
        # 每层Transformer的参数量
        # 注意力层: 3个线性层(Q,K,V) + 输出投影层
        attention_params_per_layer = 4 * d_model * d_model
        # FFN层: 2个线性层
        ffn_params_per_layer = 2 * d_model * ffn_dim
        # LayerNorm参数（RMSNorm只有scale参数）
        norm_params_per_layer = 2 * d_model
        
        transformer_params_per_layer = attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        total_transformer_params = n_layers * transformer_params_per_layer
        
        # 输出层参数
        output_params = 0 if tie_weights else vocab_size * d_model
        
        # 最终LayerNorm
        final_norm_params = d_model
        
        # 总参数量
        total_params = embedding_params + total_transformer_params + output_params + final_norm_params
        
        param_info = {
            'embedding_params': embedding_params,
            'transformer_params': total_transformer_params,
            'output_params': output_params,
            'norm_params': final_norm_params,
            'total_params': total_params,
            'vocab_size': vocab_size,
            'tie_weights': tie_weights
        }
        
        logger.info(f"📊 模型参数量估算:")
        logger.info(f"  - Embedding层: {embedding_params:,} 参数")
        logger.info(f"  - Transformer层: {total_transformer_params:,} 参数")
        logger.info(f"  - 输出层: {output_params:,} 参数 (tie_weights={tie_weights})")
        logger.info(f"  - 归一化层: {final_norm_params:,} 参数")
        logger.info(f"  - 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info(f"  - 模型大小: {total_params * 4 / 1024**2:.1f} MB (FP32)")
        
        return param_info


class FineTuningConfig(BaseConfig):
    """微调配置管理器"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.validate_config()
        
    def validate_config(self):
        """验证微调配置"""
        required_sections = ['model', 'training', 'data', 'checkpoint']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"微调配置文件缺少必需的节: {section}")
        
        # 设置微调特有的默认值
        training_config = self.config['training']
        
        # 微调通常使用较小的学习率和较少的轮数
        defaults = {
            'num_epochs': 1,
            'learning_rate': 1e-5,  # 微调使用更小的学习率
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'gradient_clip_val': 1.0,
            'use_amp': True
        }
        
        for key, default_value in defaults.items():
            if key not in training_config:
                training_config[key] = default_value
                logger.info(f"📝 设置微调配置默认值: {key} = {default_value}")


class InferenceConfig(BaseConfig):
    """推理配置管理器"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.validate_config()
        
    def validate_config(self):
        """验证推理配置"""
        # 推理配置相对简单，主要验证必需的模型和推理参数
        if 'model' not in self.config:
            raise ValueError("推理配置文件缺少模型配置节")
        
        # 设置推理默认参数
        if 'inference' not in self.config:
            self.config['inference'] = {}
        
        inference_config = self.config['inference']
        defaults = {
            'max_new_tokens': 256,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.9,
            'do_sample': True,
            'use_kv_cache': True
        }
        
        for key, default_value in defaults.items():
            if key not in inference_config:
                inference_config[key] = default_value
                logger.info(f"📝 设置推理配置默认值: {key} = {default_value}")


def create_config_manager(config_path: str, config_type: str = "pretraining") -> BaseConfig:
    """配置管理器工厂函数"""
    config_type = config_type.lower()
    
    if config_type == "pretraining":
        return PretrainingConfig(config_path)
    elif config_type == "finetuning":
        return FineTuningConfig(config_path)
    elif config_type == "inference":
        return InferenceConfig(config_path)
    else:
        # 默认返回基础配置管理器
        return BaseConfig(config_path)


def merge_configs(base_config_path: str, override_config_path: str, 
                 output_path: Optional[str] = None) -> Dict:
    """合并两个配置文件"""
    base_config = BaseConfig(base_config_path)
    override_config = BaseConfig(override_config_path)
    
    # 深度合并配置
    merged_config = _deep_merge_dict(base_config.config.copy(), override_config.config)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"✅ 合并后的配置已保存: {output_path}")
    
    return merged_config


def _deep_merge_dict(base_dict: Dict, override_dict: Dict) -> Dict:
    """深度合并字典"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    # 测试配置管理器
    import argparse
    
    parser = argparse.ArgumentParser(description="配置管理器测试")
    parser.add_argument("--config", type=str, default="pretrain/config/config.yaml", 
                       help="配置文件路径")
    parser.add_argument("--type", type=str, default="pretraining", 
                       choices=["pretraining", "finetuning", "inference"],
                       help="配置类型")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        config_manager = create_config_manager(args.config, args.type)
        print("✅ 配置管理器测试成功")
        
        if isinstance(config_manager, PretrainingConfig):
            # 测试参数量计算
            param_info = config_manager.calculate_model_parameters()
            print(f"📊 总参数量: {param_info['total_params']:,}")
            
            # 测试训练步数计算（假设数据集大小）
            steps_per_shard, steps_per_epoch = config_manager.calculate_training_steps(100000)
            print(f"📊 每分片步数: {steps_per_shard}, 每轮步数: {steps_per_epoch}")
            
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        raise
