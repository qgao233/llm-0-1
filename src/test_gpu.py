#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU支持测试脚本
"""

import torch
from llmcore import get_device, get_default_config

def test_gpu_support():
    """测试GPU支持功能"""
    print("=== GPU支持测试 ===")
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n=== 设备检测测试 ===")
    
    # 测试自动检测设备
    device_auto = get_device()
    print(f"自动检测设备: {device_auto}")
    
    # 测试强制使用CPU
    device_cpu = get_device(force_cpu=True)
    print(f"强制CPU设备: {device_cpu}")
    
    print("\n=== 配置测试 ===")
    
    # 测试默认配置
    config_auto = get_default_config()
    print(f"默认配置设备: {config_auto['device']}")
    
    # 测试强制CPU配置
    config_cpu = get_default_config(force_cpu=True)
    print(f"强制CPU配置设备: {config_cpu['device']}")
    
    print("\n=== 张量操作测试 ===")
    
    # 测试张量在不同设备上的创建
    if torch.cuda.is_available():
        print("测试GPU张量操作...")
        try:
            x_gpu = torch.randn(1000, 1000).cuda()
            y_gpu = torch.randn(1000, 1000).cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"GPU矩阵乘法成功，结果形状: {z_gpu.shape}")
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # 清理GPU内存
            del x_gpu, y_gpu, z_gpu
            torch.cuda.empty_cache()
            print("GPU内存已清理")
            
        except Exception as e:
            print(f"GPU测试失败: {e}")
    
    print("测试CPU张量操作...")
    try:
        x_cpu = torch.randn(1000, 1000)
        y_cpu = torch.randn(1000, 1000)
        z_cpu = torch.mm(x_cpu, y_cpu)
        print(f"CPU矩阵乘法成功，结果形状: {z_cpu.shape}")
    except Exception as e:
        print(f"CPU测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_gpu_support()
