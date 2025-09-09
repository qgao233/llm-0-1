from llmcore import create_model
from config_manager import get_config_manager
import torch
import torch.nn.functional as F
import json
import os
import time
import pandas as pd
import numpy as np
import urllib.request

# ==================== 数据准备函数 ====================

def download_and_prepare_data(data_file="xiyouji.txt", force_download=False):
    """下载并准备数据集"""
    if not os.path.exists(data_file) or force_download:
        print(f"正在下载数据集到 {data_file}...")
        url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
        urllib.request.urlretrieve(url, data_file)
        print("数据集下载完成")
    
    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.read()
    
    # 创建词表
    vocab = sorted(list(set(lines)))
    print(f'词表大小: {len(vocab)}')
    
    # 创建编码映射
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    # 编码整个数据集
    dataset = torch.tensor([stoi[ch] for ch in lines], dtype=torch.int16)
    print(f'数据集形状: {dataset.shape}')
    
    return dataset, vocab, itos, stoi

def encode_text(text, stoi):
    """编码文本为数字序列"""
    return [stoi[ch] for ch in text]

def decode_text(indices, itos):
    """解码数字序列为文本"""
    return ''.join([itos[i] for i in indices])

# ==================== 数据处理函数 ====================

def get_batches(data, split, batch_size, context_window, device=None):
    """构建批次数据"""
    # 切分训练集，验证集，测试集，比例为，训练80%，验证10%，测试10%
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部的训练数据作为batch，验证集，测试集也换个变量存储（单纯为了方便看）
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # 如果使用GPU，先将数据移动到GPU上，避免重复传输
    if device is not None and device.type == 'cuda' and batch_data.device != device:
        batch_data = batch_data.to(device)

    # 在GPU上生成随机索引，避免CPU到GPU的传输
    device_for_randint = device if device is not None else torch.device('cpu')
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,), device=device_for_randint)

    # 使用更高效的向量化索引方式，避免Python循环
    # 创建索引矩阵，一次性获取所有batch数据
    batch_indices = ix.unsqueeze(1) + torch.arange(context_window, device=ix.device).unsqueeze(0)
    target_indices = batch_indices + 1
    
    x = batch_data[batch_indices].long()
    y = batch_data[target_indices].long()

    # 确保数据在正确的设备上
    if device is not None:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    # 返回特征值，目标值
    return x, y

# ==================== 评估函数 ====================

@torch.no_grad()
def evaluate_loss(model, dataset, config):
    """构造一个评估函数"""
    # 评估结果存储变量
    out = {}

    # 将模型置为评估模式
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    for split in ["train", "val"]:

        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'], config.get('device'))

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out

# ==================== 优化器和调度器 ====================

def create_optimizer(model, config_manager=None, **kwargs):
    """创建优化器 - 使用配置文件或传入参数"""
    if config_manager is not None:
        opt_config = config_manager.get_optimizer_config()
    else:
        # 默认配置
        opt_config = {
            'type': 'Adam',
            'lr': 0.001,
            'betas': [0.9, 0.999],
            'weight_decay': 0.1,
            'eps': 1e-8
        }
    
    # 用传入的参数覆盖配置
    opt_config.update(kwargs)
    
    optimizer_type = opt_config.get('type', 'Adam')
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            betas=tuple(opt_config['betas']),
            weight_decay=opt_config['weight_decay'],
            eps=opt_config['eps']
        )
    else:  # 默认使用 Adam
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            betas=tuple(opt_config['betas']),
            weight_decay=opt_config['weight_decay'],
            eps=opt_config['eps']
        )

class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器，包含自适应调整功能"""
    def __init__(self, optimizer, warmup_steps=100, max_steps=10000, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        # 自适应学习率调整
        self.plateau_patience = 10  # loss停滞容忍步数
        self.plateau_factor = 0.5   # loss停滞时的学习率衰减因子
        self.best_loss = float('inf')
        self.plateau_count = 0
        
    def step(self, val_loss=None):
        self.current_step += 1
        
        # 检查是否需要自适应调整学习率
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.plateau_count = 0
            else:
                self.plateau_count += 1
                
            # 如果loss停滞，降低学习率
            if self.plateau_count >= self.plateau_patience:
                self.base_lr *= self.plateau_factor
                self.plateau_count = 0
                print(f"  📉 检测到loss停滞，学习率降低至: {self.base_lr:.2e}")
        
        if self.current_step <= self.warmup_steps:
            # 预热阶段：线性增长
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # 余弦退火阶段
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """返回调度器的状态字典"""
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'best_loss': self.best_loss,
            'plateau_count': self.plateau_count,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'min_lr': self.min_lr,
            'plateau_patience': self.plateau_patience,
            'plateau_factor': self.plateau_factor
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器的状态字典"""
        self.current_step = state_dict['current_step']
        self.base_lr = state_dict['base_lr']
        self.best_loss = state_dict['best_loss']
        self.plateau_count = state_dict['plateau_count']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.min_lr = state_dict['min_lr']
        self.plateau_patience = state_dict['plateau_patience']
        self.plateau_factor = state_dict['plateau_factor']

def create_scheduler(optimizer, config_manager=None, max_steps=10000, **kwargs):
    """创建带预热的学习率调度器"""
    if config_manager is not None:
        sch_config = config_manager.get_scheduler_config()
    else:
        # 默认配置
        sch_config = {
            'warmup_steps': 100,
            'min_lr': 1e-6,
            'plateau_patience': 10,
            'plateau_factor': 0.5
        }
    
    # 用传入的参数覆盖配置
    sch_config.update(kwargs)
    
    return WarmupCosineScheduler(
        optimizer, 
        warmup_steps=sch_config['warmup_steps'],
        max_steps=max_steps,
        min_lr=sch_config['min_lr']
    )

# ==================== 训练函数 ====================

def train(model, optimizer, dataset, config, scheduler=None, print_logs=False):
    """构建训练函数"""
    # loss存储
    losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 50  # 早停耐心值

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'], config.get('device'))

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改
        if scheduler:
            # 每个batch都更新学习率
            scheduler.step()

        # 打印log
        if epoch % config['log_interval'] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model, dataset, config)

            # Store the validation loss
            losses += [x]
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 打印进度日志
            if print_logs:
                print(f"Epoch {epoch:5d} | train loss {x['train']:.4f} | val loss {x['val']:.4f} | lr {current_lr:.2e} | Time {batch_time:.2f}s | ETA {batch_time * (config['epochs'] - epoch)/config['log_interval']:.1f}s")

            # 将验证loss传递给调度器进行自适应调整
            if scheduler and hasattr(scheduler, 'step'):
                # 对于自定义调度器，传递验证loss
                if hasattr(scheduler, 'plateau_patience'):
                    scheduler.step(val_loss=x['val'])
            
            # 早停机制
            if x['val'] < best_val_loss:
                best_val_loss = x['val']
                patience_counter = 0
                if print_logs:
                    print(f"  ✓ 新的最佳验证loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  ⚠ 早停触发！验证loss连续{max_patience}轮未改善")
                    break

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

    # 上面所有epoch训练结束，打印最终的结果
    print(f"\n=== 训练结束 ===")
    print(f"最佳验证loss: {best_val_loss:.4f}")
    print(f"最终验证loss: {losses[-1]['val']:.4f}")

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()

# ==================== 训练环境初始化 ====================

def initialize_training_environment(config_path="../config.yaml", config_manager=None):
    """初始化训练环境，返回所有必要的组件"""
    if config_manager is None:
        config_manager = get_config_manager(config_path)
    
    # 获取配置
    config = config_manager.get_training_config()
    data_config = config_manager.get_data_config()
    
    # 准备数据
    dataset, vocab, itos, stoi = download_and_prepare_data(
        data_file=data_config.get('data_file', 'xiyouji.txt'),
        force_download=data_config.get('force_download', False)
    )
    
    # 更新配置中的词表大小
    config['vocab_size'] = len(vocab)
    
    # 如果使用GPU，将数据集移动到GPU上以避免重复传输
    device = config.get('device', torch.device('cpu'))
    if device.type == 'cuda':
        print(f"将数据集移动到GPU: {device}")
        dataset = dataset.to(device)
        # 预热GPU，减少首次运行的开销
        torch.cuda.synchronize()
        print("GPU预热完成")
    
    # 创建模型
    model = create_model(config)
    
    return {
        'model': model,
        'dataset': dataset,
        'vocab': vocab,
        'itos': itos,
        'stoi': stoi,
        'config': config,
        'config_manager': config_manager
    }

# ==================== GPU优化和辅助函数 ====================

def setup_gpu_optimization():
    """设置GPU优化配置"""
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
        return True
    return False

def print_gpu_memory_info(device):
    """打印GPU内存信息"""
    if device.type != 'cuda':
        return
        
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
    
def print_device_info(device):
    """打印设备信息"""
    print(f"当前使用的计算设备: {device}")
    if device.type == 'cuda':
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print_gpu_memory_info(device)

def save_model_and_config(model, config, optimizer, scheduler, save_dir="./model_save"):
    """保存模型、配置和训练状态"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("保存模型...")

# 保存模型权重
    model_save_path = os.path.join(save_dir, "pytorch_model.bin")
    state_dict = model.state_dict()
    # 过滤掉缓存，只保存真正的模型参数
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.R_cache')}
    torch.save(filtered_state_dict, model_save_path)
    
    # 保存配置文件
    config_save_path = os.path.join(save_dir, "config.json")
config_serializable = {k: v for k, v in config.items() if k != 'device'}
config_serializable['device_type'] = str(config['device'])  # 保存设备类型的字符串表示
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)

    # 保存优化器状态
    optimizer_save_path = os.path.join(save_dir, "optimizer.pt")
torch.save(optimizer.state_dict(), optimizer_save_path)

    # 保存调度器状态
    if scheduler is not None:
        scheduler_save_path = os.path.join(save_dir, "scheduler.pt")
torch.save(scheduler.state_dict(), scheduler_save_path)

    print(f"训练完成，模型已保存到 {save_dir}")

def print_final_gpu_stats(device):
    """打印最终GPU统计信息"""
    if device.type != 'cuda':
        return
        
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

def run_training(config_path="../config.yaml", use_scheduler=None):
    """运行训练流程"""
    try:
        # 设置GPU优化
        gpu_available = setup_gpu_optimization()
        
        # 初始化训练环境
        print("正在初始化训练环境...")
        env = initialize_training_environment(config_path)
        
        model = env['model']
        dataset = env['dataset']
        config = env['config']
        config_manager = env['config_manager']
        
        # 显示设备信息
        device = config['device']
        print_device_info(device)
        
        # 打印配置信息
        print("\n📋 当前训练配置:")
        config_manager.print_config()
        
        # 创建优化器
        optimizer = create_optimizer(model, config_manager)
        
        # 创建调度器
        scheduler = None
        scheduler_config = config_manager.get_scheduler_config()
        
        if use_scheduler is None:
            use_scheduler = scheduler_config.get('enabled', False)
        
        if use_scheduler:
            # 根据总训练步数设置调度器参数
            total_steps = config['epochs']
            scheduler = create_scheduler(optimizer, config_manager, max_steps=total_steps)
            
            print(f"调度器配置:")
            print(f"  预热步数: {scheduler_config.get('warmup_steps', 100)}")
            print(f"  总训练步数: {total_steps}")
        
        print(f"\n优化器配置:")
        print(f"  类型: {config_manager.get_optimizer_config().get('type', 'Adam')}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  使用调度器: {'是' if scheduler else '否'}")
        
        # 开始训练
        print("\n开始正式训练...")
        start_training_time = time.time()
        train(model, optimizer, dataset, config, scheduler=scheduler, print_logs=True)
        training_time = time.time() - start_training_time
        
        print(f"\n=== 训练完成 ===")
        print(f"总训练时间: {training_time:.2f}秒")
        if config['epochs'] > 0:
            print(f"平均每epoch时间: {training_time/config['epochs']:.4f}秒")
        
        # 保存模型和配置
        save_model_and_config(model, config, optimizer, scheduler)
        
        # 显示最终GPU内存使用情况
        print_final_gpu_stats(device)
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 启动模型训练...")
    print("=" * 60)
    
    # 可以通过命令行参数控制配置和调度器
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='训练LLM模型')
    parser.add_argument('--config', default='../config.yaml', help='配置文件路径')
    parser.add_argument('--scheduler', action='store_true', help='强制启用学习率调度器')
    parser.add_argument('--no-scheduler', action='store_true', help='强制禁用学习率调度器')
    
    args = parser.parse_args()
    
    # 确定是否使用调度器
    use_scheduler = None
    if args.scheduler:
        use_scheduler = True
        print("📊 强制启用学习率调度器")
    elif args.no_scheduler:
        use_scheduler = False
        print("📊 强制禁用学习率调度器")
    else:
        print("📊 根据配置文件决定是否使用调度器")
    
    print(f"📂 使用配置文件: {args.config}")
    
    success = run_training(config_path=args.config, use_scheduler=use_scheduler)
    
    if success:
        print("\n🎉 训练成功完成！")
        print("💡 现在可以运行以下命令:")
        print("  - python src/infer.py    # 推理演示")
        print("  - python src/chat.py     # 聊天模式")
        print("💡 配置文件相关命令:")
        print("  - python src/config_manager.py  # 查看配置")
        print("  - 直接编辑 config.yaml 文件修改超参数")
    else:
        print("\n❌ 训练失败，请检查错误信息")
        sys.exit(1)