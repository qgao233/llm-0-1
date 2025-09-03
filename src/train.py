from llmcore import (
    initialize_training_environment, 
    create_optimizer, 
    create_scheduler, 
    train,
    get_default_config,
    get_batches
)
import torch
import json
import os
import time


# 优化GPU设置
if torch.cuda.is_available():
    print("正在配置GPU优化设置...")
    # 清空GPU缓存
    torch.cuda.empty_cache()
    # 设置使用95%的GPU内存
    torch.cuda.set_per_process_memory_fraction(0.95)
    # 启用cudnn自动调优，首次运行会慢但后续会快
    torch.backends.cudnn.benchmark = True
    # 启用确定性操作（可选，会稍微影响性能但保证可重现性）
    # torch.backends.cudnn.deterministic = True
    print("GPU优化设置完成")

# 初始化训练环境
print("正在初始化训练环境...")
env = initialize_training_environment()

model = env['model']
dataset = env['dataset']
config = env['config']

# 显示设备信息
device = config['device']
print(f"当前使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 显示详细的GPU内存信息
    def print_gpu_memory_info():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved
        print(f"GPU内存状态:")
        print(f"  已分配: {allocated:.3f} GB")
        print(f"  已预留: {reserved:.3f} GB") 
        print(f"  可用: {free:.3f} GB")
        print(f"  总计: {total:.3f} GB")
        print(f"  使用率: {reserved/total*100:.1f}%")
    
    print_gpu_memory_info()

# 创建优化器和调度器
optimizer = create_optimizer(model)
# 根据总训练步数设置调度器参数
total_steps = config['epochs']
warmup_steps = min(100, total_steps // 10)  # 预热步数为总步数的10%，最多100步
scheduler = create_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_steps)

print(f"优化器配置:")
print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
print(f"  预热步数: {warmup_steps}")
print(f"  总训练步数: {total_steps}")

# 开始训练
print("\n开始正式训练...")
start_training_time = time.time()
# train(model, optimizer, dataset, config, scheduler=scheduler, print_logs=True)
train(model, optimizer, dataset, config, scheduler=None, print_logs=True)
training_time = time.time() - start_training_time

print(f"\n=== 训练完成 ===")
print(f"总训练时间: {training_time:.2f}秒")
print(f"平均每epoch时间: {training_time/config['epochs']:.4f}秒")


# 保存模型权重
model_save_dir = "./model_save"
os.makedirs(model_save_dir, exist_ok=True)

print("保存模型...")
model_save_path = f"{model_save_dir}/pytorch_model.bin"
# 过滤掉缓存，只保存真正的模型参数
state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.R_cache')}
torch.save(filtered_state_dict, model_save_path)

# 生成一个config文件（移除不可序列化的device对象）
config_save_path = f"{model_save_dir}/config.json"
config_serializable = {k: v for k, v in config.items() if k != 'device'}
config_serializable['device_type'] = str(config['device'])  # 保存设备类型的字符串表示
with open(config_save_path, 'w') as f:
    json.dump(config_serializable, f, indent=2)

# 保存optimizer和学习率调度器的状态，方便继续微调
optimizer_save_path = f"{model_save_dir}/optimizer.pt"
torch.save(optimizer.state_dict(), optimizer_save_path)

scheduler_save_path = f"{model_save_dir}/scheduler.pt"
torch.save(scheduler.state_dict(), scheduler_save_path)

print("训练完成，模型已保存到", model_save_dir)

# 显示最终GPU内存使用情况
if device.type == 'cuda':
    print("\n=== 训练完成后的GPU内存状态 ===")
    final_allocated = torch.cuda.memory_allocated(device) / 1024**3
    final_reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    print(f"最终已分配内存: {final_allocated:.3f} GB")
    print(f"最终已预留内存: {final_reserved:.3f} GB")
    print(f"最大已分配内存: {max_allocated:.3f} GB")
    print(f"GPU总内存: {total:.3f} GB")
    print(f"峰值使用率: {max_allocated/total*100:.1f}%")