#!/usr/bin/env python3
"""
配置管理模块 - 加载和管理 YAML 配置文件
"""

import yaml
import os
import torch
from typing import Dict, Any

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"⚠️ 配置文件 {self.config_path} 不存在，使用默认配置")
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            print("使用默认配置")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'training': {
                'batch_size': 64,
                'epochs': 8000,
                'log_interval': 20
            },
            'model': {
                'context_window': 64,
                'vocab_size': 4325,
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6,
                'ffn_dim': 1024,
                'dropout': 0.1
            },
            'data': {
                'data_file': 'xiyouji.txt',
                'force_download': False
            },
            'device': {
                'force_cpu': False
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.1,
                'eps': 1e-8
            },
            'scheduler': {
                'enabled': False,
                'warmup_steps': 100,
                'min_lr': 1e-6,
                'plateau_patience': 10,
                'plateau_factor': 0.5
            },
            'inference': {
                'max_new_tokens': 100,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'enable_warmup': True
            },
            'chat': {
                'max_tokens': 150,
                'stream_mode': True,
                'context_length': 5
            }
        }
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        force_cpu = self.config.get('device', {}).get('force_cpu', False)
        
        if force_cpu:
            device = torch.device("cpu")
            print("强制使用CPU进行计算")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("CUDA不可用，使用CPU进行计算")
        
        return device
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        training_config = self.config['training'].copy()
        training_config.update(self.config['model'])
        training_config['device'] = self.get_device()
        return training_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        model_config = self.config['model'].copy()
        model_config['device'] = self.get_device()
        return model_config
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config['optimizer'].copy()
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置"""
        return self.config['scheduler'].copy()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.config['inference'].copy()
    
    def get_chat_config(self) -> Dict[str, Any]:
        """获取聊天配置"""
        return self.config['chat'].copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config['data'].copy()
    
    def save_config(self, output_path: str = None):
        """保存当前配置到文件"""
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"✅ 配置已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
    def update_config(self, section: str, key: str, value: Any):
        """更新配置项"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        print(f"📝 配置已更新: {section}.{key} = {value}")
    
    def print_config(self):
        """打印当前配置"""
        print("📋 当前配置:")
        print("=" * 50)
        for section, items in self.config.items():
            print(f"[{section}]")
            if isinstance(items, dict):
                for key, value in items.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {items}")
            print()

def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager(config_path)

# 向后兼容的函数
def get_default_config(force_cpu=False, config_path="config.yaml"):
    """获取默认配置 - 向后兼容"""
    config_manager = ConfigManager(config_path)
    if force_cpu:
        config_manager.update_config('device', 'force_cpu', True)
    return config_manager.get_training_config()

if __name__ == "__main__":
    # 测试配置管理器
    print("🧪 测试配置管理器...")
    
    config_manager = ConfigManager()
    config_manager.print_config()
    
    print("\n🎯 获取各模块配置:")
    print(f"训练配置: {config_manager.get_training_config()}")
    print(f"推理配置: {config_manager.get_inference_config()}")
    print(f"聊天配置: {config_manager.get_chat_config()}")
