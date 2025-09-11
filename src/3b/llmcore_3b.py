#!/usr/bin/env python3
"""
3B参数LLM模型核心实现
基于Transformer架构，针对30亿参数规模优化
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from collections import OrderedDict
from typing import Optional, Tuple

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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - 3B模型优化版本"""
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
        self.eps = eps

    def forward(self, x):
        # 使用更高效的RMSNorm实现
        ffn_layernorm = x.pow(2).mean(-1, keepdim=True)
        ffn_layernorm = ffn_layernorm + self.eps
        ffn_layernorm = ffn_layernorm.sqrt()
        ffn_layernorm = x / ffn_layernorm
        return self.scale * ffn_layernorm


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 - 支持更长的上下文窗口"""
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 预计算旋转矩阵
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len):
        seq = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMaskedAttentionHead(nn.Module):
    """带旋转位置编码的注意力头 - 3B模型优化版本"""
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.head_dim = self.d_model // self.n_heads
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # 线性变换层 - 每个头只处理自己的维度
        self.w_q = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.w_v = nn.Linear(self.d_model, self.head_dim, bias=False)
        
        # 旋转位置编码
        self.rope = RotaryPositionalEmbedding(self.head_dim, config['context_window'])
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 注册因果掩码
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config['context_window'], config['context_window']))
            .view(1, 1, config['context_window'], config['context_window'])
        )

    def forward(self, x, use_kv_cache=False, past_kv=None):
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换 - 每个头输出head_dim维度
        q = self.w_q(x).view(batch_size, seq_len, self.head_dim).unsqueeze(1)  # (B, 1, S, head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.head_dim).unsqueeze(1)  # (B, 1, S, head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.head_dim).unsqueeze(1)  # (B, 1, S, head_dim)
        
        # 应用旋转位置编码
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV缓存支持（用于推理加速）
        if use_kv_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码
        mask_len = scores.shape[-1]
        scores = scores.masked_fill(
            self.mask[:, :, :seq_len, :mask_len] == 0, float('-inf')
        )
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        out = torch.matmul(attn_weights, v)  # (B, 1, S, head_dim)
        out = out.squeeze(1)  # (B, S, head_dim)
        
        # 返回输出和KV缓存
        if use_kv_cache:
            return out, (k, v)
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力 - 3B模型版本"""
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # 修正：每个头的输出维度是 d_model // n_heads，不是 d_model
        head_dim = config['d_model'] // config['n_heads']
        self.linear = nn.Linear(config['n_heads'] * head_dim, config['d_model'])

    def forward(self, x, use_kv_cache=False, past_kvs=None):
        if use_kv_cache and past_kvs is not None:
            head_outputs = []
            new_kvs = []
            for i, head in enumerate(self.heads):
                out, kv = head(x, use_kv_cache=True, past_kv=past_kvs[i] if past_kvs else None)
                head_outputs.append(out)
                new_kvs.append(kv)
            return self.linear(torch.cat(head_outputs, dim=-1)), new_kvs
        else:
            head_outputs = [head(x) for head in self.heads]
            return self.linear(torch.cat(head_outputs, dim=-1))


class SwiGLU(nn.Module):
    """SwiGLU激活函数 - 更适合大模型"""
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU激活
        up = self.w_up(x)
        return self.w_down(gate * up)


class LlmpvqBlock(nn.Module):
    """Transformer块 - 3B模型版本"""
    def __init__(self, config):
        super().__init__()
        
        # 注意力层
        self.attention = MultiHeadAttention(config)
        self.rms_1 = RMSNorm(config['d_model'])
        
        # FFN层
        ffn_dim = config.get('ffn_dim', config['d_model'] * 4)
        self.ffn = SwiGLU(config['d_model'], ffn_dim)
        self.rms_2 = RMSNorm(config['d_model'])

        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

    def forward(self, x, use_kv_cache=False, past_kv=None):
        # 注意力层（Pre-Norm）
        normed_x = self.rms_1(x)
        if use_kv_cache:
            attn_out, new_kv = self.attention(normed_x, use_kv_cache=True, past_kvs=past_kv)
        else:
            attn_out = self.attention(normed_x)
            new_kv = None
        
        x = x + self.dropout(attn_out)
        
        # FFN层（Pre-Norm）
        normed_x = self.rms_2(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)
        
        if use_kv_cache:
            return x, new_kv
        return x


class Llmpvq3B(nn.Module):
    """3B参数的LLM模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])

        # Transformer层
        self.layers = nn.ModuleDict(
            OrderedDict([(f"Llmpvq3B_{i}", LlmpvqBlock(config)) for i in range(config['n_layers'])])
        )
        
        # 最终归一化层
        self.final_norm = RMSNorm(config['d_model'])

        # 输出层
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        # 权重共享（可选）
        if config.get('tie_weights', True):
            self.lm_head.weight = self.embeddings.weight

        # 初始化权重
        self.apply(self._init_weights)
        # 计算参数量
        self.print_model_info()

    def _init_weights(self, module):
        """权重初始化 - 针对大模型优化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n🏗️  3B模型架构信息:")
        print(f"  📊 总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  🎯 可训练参数: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
        print(f"  🔧 模型配置:")
        print(f"    - d_model: {self.config['d_model']}")
        print(f"    - n_heads: {self.config['n_heads']}")
        print(f"    - n_layers: {self.config['n_layers']}")
        print(f"    - vocab_size: {self.config['vocab_size']}")
        print(f"    - context_window: {self.config['context_window']}")
        print(f"    - ffn_dim: {self.config.get('ffn_dim', self.config['d_model'] * 4)}")

    def forward(self, idx, use_kv_cache=False, past_kvs=None):
        batch_size, seq_len = idx.shape
        
        # 嵌入
        x = self.embeddings(idx)  # (batch_size, seq_len, d_model)
        
        # Transformer层
        new_kvs = [] if use_kv_cache else None
        for i, (name, layer) in enumerate(self.layers.items()):
            if use_kv_cache:
                x, new_kv = layer(x, use_kv_cache=True, 
                                past_kv=past_kvs[i] if past_kvs else None)
                new_kvs.append(new_kv)
            else:
                x = layer(x)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 输出投影
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        if use_kv_cache:
            return logits, new_kvs
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_kv_cache=True):
        """生成文本 - 支持KV缓存的高效推理"""
        self.eval()
        past_kvs = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 如果使用KV缓存，只需要处理最后一个token
                if use_kv_cache and past_kvs is not None:
                    idx_input = idx[:, -1:]
                else:
                    idx_input = idx[:, -self.config['context_window']:]
                
                # 前向传播
                if use_kv_cache:
                    logits, past_kvs = self(idx_input, use_kv_cache=True, past_kvs=past_kvs)
                else:
                    logits = self(idx_input)
                
                # 获取最后一个位置的logits
                logits = logits[:, -1, :] / temperature
                
                # Top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Top-p采样
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # 采样
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # 添加到序列
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_3b_model(config):
    """创建3B模型实例"""
    return Llmpvq3B(config)


if __name__ == "__main__":
    # 测试3B模型
    config = {
        'context_window': 2048,
        'vocab_size': 50000,
        'd_model': 2304,
        'n_heads': 32,
        'n_layers': 32,
        'ffn_dim': 9216,
        'dropout': 0.1,
        'tie_weights': True
    }
    
    print("🚀 创建3B模型...")
    model = create_3b_model(config)
    
    # 测试前向传播
    batch_size, seq_len = 2, 512
    test_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"\n🧪 测试前向传播...")
    print(f"输入形状: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
        print(f"输出形状: {output.shape}")
        print(f"✅ 3B模型测试成功！")
