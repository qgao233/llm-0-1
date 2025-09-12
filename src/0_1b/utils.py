#!/usr/bin/env python3
"""
工具函数模块
包含设备检测和其他通用工具函数
"""

import torch


def get_device(force_cpu=False, device_config=None):
    """检测并返回可用的设备，支持多GPU - 3B模型版本"""
    if force_cpu:
        device = torch.device("cpu")
        print("⚠️  强制使用CPU进行计算 - 注意：3B模型在CPU上会非常慢")
        return device, False, []
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("❌ CUDA不可用，使用CPU进行计算 - 警告：3B模型需要大量内存和时间")
        return device, False, []
    
    # 获取可用的GPU数量和信息
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个可用GPU:")
    
    available_gpus = []
    total_memory = 0
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        available_gpus.append(i)
        total_memory += gpu_memory
    
    print(f"📊 总GPU内存: {total_memory:.1f} GB")
    
    # 3B模型内存需求检查
    estimated_memory_gb = 12  # 3B模型大约需要12GB内存（FP16）
    if total_memory < estimated_memory_gb:
        print(f"⚠️  警告: 3B模型预计需要约{estimated_memory_gb}GB内存，当前总内存{total_memory:.1f}GB可能不足")
        print("💡 建议启用混合精度训练和梯度检查点来节省内存")
    
    # 检查是否启用多GPU训练
    use_multi_gpu = gpu_count > 1
    if device_config and 'multi_gpu' in device_config:
        use_multi_gpu = device_config['multi_gpu'] and gpu_count > 1
    
    # 选择主GPU设备
    primary_device = torch.device("cuda:0")
    
    if use_multi_gpu:
        print(f"🚀 启用多GPU训练，使用 {gpu_count} 个GPU - 推荐用于3B模型")
        return primary_device, True, available_gpus
    else:
        print(f"📱 使用单GPU训练: {torch.cuda.get_device_name(0)}")
        return primary_device, False, [0]
