from llmcore import (
    create_model, 
    generate, 
    download_and_prepare_data,
    get_device
)
import torch
import json
import os

# 加载配置
config_path = "./model_save/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# 确保使用正确的设备
if 'device' not in config and 'device_type' not in config:
    config['device'] = get_device()
else:
    # 从device_type重建device对象
    if 'device_type' in config:
        device_str = config['device_type']
        if 'cuda' in device_str and torch.cuda.is_available():
            config['device'] = torch.device('cuda')
        else:
            config['device'] = torch.device('cpu')
    elif 'device' in config:
        # 兼容旧版本配置
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            print("警告: 配置文件指定使用CUDA，但CUDA不可用，改用CPU")
            config['device'] = torch.device('cpu')
        elif isinstance(config['device'], str):
            config['device'] = torch.device(config['device'])

print("加载模型配置:", config)
print(f"推理设备: {config['device']}")

# 创建模型
model = create_model(config)

# 加载模型权重
model_save_path = "./model_save/pytorch_model.bin"
print("加载模型权重...")
# 根据当前设备加载模型权重
device = config['device']
if device.type == 'cuda':
    state_dict = torch.load(model_save_path)
else:
    # 如果当前使用CPU但模型是在GPU上训练的，需要映射到CPU
    state_dict = torch.load(model_save_path, map_location='cpu')

# 过滤掉缓存相关的键，这些不是模型参数
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.R_cache')}

# 加载过滤后的状态字典
missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

if missing_keys:
    print(f"警告: 缺少的键: {missing_keys}")
if unexpected_keys:
    print(f"警告: 意外的键: {unexpected_keys}")

print("模型权重加载完成")

# 设置为评估模式
model.eval()
print(f"模型已加载到设备: {device}")

# 准备数据（主要是为了获取词表）
print("准备词表...")
dataset, vocab, itos, stoi = download_and_prepare_data()

# 进行推理
print("开始推理...")
output = generate(model, itos, config, max_new_tokens=50)
print("生成结果:")
for i, text in enumerate(output):
    print(f"序列 {i+1}: {text}")
