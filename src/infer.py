from llmcore import (
    create_model, 
    generate, 
    download_and_prepare_data
)
import torch
import json
import os

# 加载配置
config_path = "./model_save/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print("加载模型配置:", config)

# 创建模型
model = create_model(config)

# 加载模型权重
model_save_path = "./model_save/pytorch_model.bin"
print("加载模型权重...")
model.load_state_dict(torch.load(model_save_path))

# 设置为评估模式
model.eval()

# 准备数据（主要是为了获取词表）
print("准备词表...")
dataset, vocab, itos, stoi = download_and_prepare_data()

# 进行推理
print("开始推理...")
output = generate(model, itos, config, max_new_tokens=50)
print("生成结果:")
for i, text in enumerate(output):
    print(f"序列 {i+1}: {text}")
