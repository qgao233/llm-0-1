from llmcore import (
    create_model, 
    get_device,
    decode_text
)
from train import download_and_prepare_data
from config_manager import get_config_manager
import torch
import torch.nn.functional as F
import json
import os

def top_k_sampling(logits, k=50):
    """Top-K采样：只保留概率最高的k个token"""
    if k <= 0:
        return logits
    
    # 获取top-k的值和索引
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # 创建一个与logits同样大小的张量，填充为负无穷
    filtered_logits = torch.full_like(logits, float('-inf'))
    
    # 只保留top-k的值
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
    
    return filtered_logits

def top_p_sampling(logits, p=0.9):
    """Top-P (nucleus)采样：保留累积概率达到p的最小token集合"""
    if p >= 1.0:
        return logits
    
    # 按概率降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # 计算softmax概率
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cumulative_probs > p
    
    # 保留第一个超过p的token（确保至少有一个token）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 创建过滤后的logits
    filtered_logits = sorted_logits.clone()
    filtered_logits[sorted_indices_to_remove] = float('-inf')
    
    # 将结果映射回原始顺序
    original_logits = torch.full_like(logits, float('-inf'))
    original_logits.scatter_(-1, sorted_indices, filtered_logits)
    
    return original_logits

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    使用temperature、top-k和top-p采样下一个token
    
    Args:
        logits: 模型输出的logits
        temperature: 温度参数，控制随机性 (0.1-2.0)
        top_k: top-k采样参数，0表示不使用
        top_p: top-p采样参数，1.0表示不使用
    """
    # 应用温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # 应用top-k采样
    if top_k > 0:
        logits = top_k_sampling(logits, top_k)
    
    # 应用top-p采样
    if top_p < 1.0:
        logits = top_p_sampling(logits, top_p)
    
    # 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 采样下一个token
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

def generate(model, itos, config, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9, seed_text=None):
    """
    生成文本 - 优化版本
    
    Args:
        model: 训练好的模型
        itos: 索引到字符的映射
        config: 配置字典
        max_new_tokens: 生成的最大token数
        temperature: 温度参数，控制随机性 (0.1-2.0)
        top_k: top-k采样参数，0表示不使用
        top_p: top-p采样参数，1.0表示不使用
        seed_text: 种子文本，如果为None则随机生成
    """
    device = config.get('device', torch.device('cpu'))
    
    # 确保模型在正确的设备上且处于评估模式
    model = model.to(device)
    model.eval()
    
    if seed_text is not None:
        # 使用提供的种子文本
        print(f"使用种子文本: '{seed_text}'")
        # 这里需要stoi来编码种子文本，暂时使用随机初始化
        idx = torch.randint(1, min(4000, config.get('vocab_size', 4000)), (1, len(seed_text))).long().to(device)
    else:
        # 生成随机数，作为输入数据
        idx = torch.randint(1, min(4000, config.get('vocab_size', 4000)), (5, 1)).long().to(device)
        print("随机生成的字符:", [itos[i.item()] for i in idx])
    
    print(f"采样参数: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print("初始序列形状:", idx.shape)
    
    # 启用推理优化
    if device.type == 'cuda':
        # 使用自动混合精度加速推理
        torch.backends.cudnn.benchmark = True
    
    # 生成循环
    generated_count = 0
    for step in range(max_new_tokens):
        # 获取当前上下文窗口的输入
        context = idx[:, -config['context_window']:]
        
        # 前向传播获取logits
        with torch.no_grad():
            logits = model(context)
        
        # 获取最后一个时间步的logits
        last_time_step_logits = logits[:, -1, :]
        
        # 使用改进的采样方法
        idx_next = sample_next_token(
            last_time_step_logits, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        
        # 拼接新生成的token
        idx = torch.cat([idx, idx_next], dim=-1)
        generated_count += 1
        
        # 检查是否生成了自然的停止符
        new_char = itos.get(idx_next[0].item(), '')
        if new_char in ['。', '！', '？', '\n'] and generated_count >= 20:
            # 生成了句号等结束符，且已生成足够长度，可以停止
            break
        
        # 可选：打印生成进度
        if step % 20 == 0 and step > 0:
            print(f"生成进度: {step}/{max_new_tokens}")
    
    # 解码生成的序列
    print("最终序列形状:", idx.shape)
    generated_texts = [decode_text(x, itos) for x in idx.tolist()]
    
    return generated_texts

def warmup_model(model, config, warmup_steps=3):
    """模型预热，减少首次推理延迟"""
    print("🔥 正在预热模型...")
    device = config['device']
    
    # 创建虚拟输入进行预热
    dummy_input = torch.randint(
        1, min(100, config.get('vocab_size', 4000)), 
        (1, config['context_window']), 
        device=device
    )
    
    model.eval()
    with torch.no_grad():
        for i in range(warmup_steps):
            # 前向传播预热
            _ = model(dummy_input)
            
            # 如果是GPU，同步一下确保操作完成
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    print("✅ 模型预热完成")

def load_model_and_config(model_dir="./model_save", enable_warmup=True):
    """加载模型和配置"""
    # 加载配置
    config_path = os.path.join(model_dir, "config.json")
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
    model_save_path = os.path.join(model_dir, "pytorch_model.bin")
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

    # 模型预热
    if enable_warmup:
        warmup_model(model, config)

    return model, config

def run_inference_demo(config_path="config/config.yaml"):
    """运行推理演示"""
    try:
        # 加载配置管理器
        config_manager = get_config_manager(config_path)
        inference_config = config_manager.get_inference_config()
        
        # 加载模型
        model, config = load_model_and_config(enable_warmup=inference_config.get('enable_warmup', True))
        
        # 准备数据（主要是为了获取词表）
        print("准备词表...")
        data_config = config_manager.get_data_config()
        dataset, vocab, itos, stoi = download_and_prepare_data(
            data_file=data_config.get('data_file', 'xiyouji.txt'),
            force_download=data_config.get('force_download', False)
        )

        # 进行推理
        print("开始推理...")
        print(f"📂 使用配置文件: {config_path}")
        
        # 从配置文件获取默认推理参数
        default_max_tokens = inference_config.get('max_new_tokens', 100)
        default_temperature = inference_config.get('temperature', 0.8)
        default_top_k = inference_config.get('top_k', 50)
        default_top_p = inference_config.get('top_p', 0.9)

        # 测试不同的采样策略 - 增加输出长度
        sampling_configs = [
            {"name": "贪婪采样", "temperature": 0.1, "top_k": 1, "top_p": 1.0, "max_tokens": 80},
            {"name": "配置文件默认", "temperature": default_temperature, "top_k": default_top_k, "top_p": default_top_p, "max_tokens": default_max_tokens},
            {"name": "创意采样", "temperature": 1.2, "top_k": 100, "top_p": 0.95, "max_tokens": 120},
            {"name": "随机采样", "temperature": 1.5, "top_k": 0, "top_p": 1.0, "max_tokens": 80}
        ]

        for config_item in sampling_configs:
            print(f"\n=== {config_item['name']} ===")
            output = generate(
                model, itos, config, 
                max_new_tokens=config_item['max_tokens'],
                temperature=config_item['temperature'],
                top_k=config_item['top_k'],
                top_p=config_item['top_p']
            )
            print("生成结果:")
            for i, text in enumerate(output):
                print(f"序列 {i+1}: {text}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("💡 请确保已经训练了模型并保存在 ./model_save/ 目录下")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行模型推理演示')
    parser.add_argument('--config', default='config/config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    print("🚀 启动模型推理演示...")
    print(f"📂 使用配置文件: {args.config}")
    run_inference_demo(config_path=args.config)
    print("🎉 推理演示完成！")
