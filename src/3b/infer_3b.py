#!/usr/bin/env python3
"""
3B参数LLM模型推理脚本
支持高效推理、KV缓存、批量生成等功能
"""

import torch
import torch.nn.functional as F
import os
import sys
import time
import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from llmcore_3b import create_3b_model, get_device
from data_processor import DataProcessor


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, 
                     top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    """采样下一个token - 3B模型优化版本"""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    logits = logits / temperature
    
    # Top-k采样
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)
    
    # Top-p采样
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 创建掩码
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # 采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LLMInference3B:
    """3B模型推理器"""
    
    def __init__(self, model_dir: str = "model_save_3b", config_path: str = "config/config.yaml"):
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        self.model = None
        self.config = None
        self.data_processor = None
        self.device = None
        self.itos = None
        self.stoi = None
        
        # 加载配置
        self._load_config()
        
        # 设置设备
        self._setup_device()
        
        # 加载模型
        self._load_model()
        
        # 准备分词器
        self._setup_tokenizer()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _setup_device(self):
        """设置推理设备"""
        device_config = self.config.get('device', {})
        self.device, is_multi_gpu, available_gpus = get_device(
            force_cpu=device_config.get('force_cpu', False),
            device_config=device_config
        )
        print(f"🎯 推理设备: {self.device}")
    
    def _load_model(self):
        """加载训练好的模型"""
        print("🔄 加载3B模型...")
        
        # 查找最佳模型文件
        best_model_path = self.model_dir / "best_model.pt"
        if not best_model_path.exists():
            # 查找最新的检查点
            checkpoints = list(self.model_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"在 {self.model_dir} 中未找到模型文件")
            
            # 按修改时间排序，选择最新的
            best_model_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"📁 使用最新检查点: {best_model_path}")
        else:
            print(f"💎 使用最佳模型: {best_model_path}")
        
        # 加载检查点
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # 获取模型配置
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
            else:
                model_config = self.config['model']
            
            # 创建模型
            self.model = create_3b_model(model_config)
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # 处理DataParallel包装的模型
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint)
            
            # 移动到设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 模型加载成功")
            if 'epoch' in checkpoint:
                print(f"📊 训练轮次: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"📉 训练损失: {checkpoint['loss']:.4f}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _setup_tokenizer(self):
        """设置分词器"""
        print("🔤 准备分词器...")
        
        data_config = self.config['data']
        tokenizer_type = "chinese_bpe" if data_config.get('use_bpe_tokenizer', True) else "char_level"
        vocab_path = data_config.get('vocab_cache_path', 'chinese_tokenizer_vocab_50k.json')
        
        from data_collectors import DataProcessor
        
        self.data_processor = DataProcessor(
            tokenizer_type=tokenizer_type,
            vocab_path=vocab_path,
            vocab_size=self.config['model']['vocab_size']
        )
        
        # 准备词汇表（如果需要的话）
        try:
            # 尝试从现有的词汇表文件加载
            if os.path.exists(vocab_path):
                print(f"✅ 分词器准备完成，词汇表大小: {self.data_processor.vocab_size or 'Unknown'}")
            else:
                print("⚠️ 词汇表文件不存在，可能需要先训练模型")
            
            self.itos = self.data_processor.itos
            self.stoi = self.data_processor.stoi
        except Exception as e:
            print(f"❌ 分词器准备失败: {e}")
            raise
    
    def encode_text(self, text: str) -> List[int]:
        """编码文本为token序列"""
        if self.data_processor:
            return self.data_processor.encode_text(text)
        else:
            return [self.stoi.get(ch, 0) for ch in text if ch in self.stoi]
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """解码token序列为文本"""
        if self.data_processor:
            return self.data_processor.decode_text(tokens)
        else:
            return ''.join([self.itos.get(idx, '') for idx in tokens])
    
    def generate(self, prompt: str, max_new_tokens: int = 512, 
                temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                use_kv_cache: bool = True, show_progress: bool = False) -> str:
        """生成文本"""
        # 编码输入
        input_tokens = self.encode_text(prompt)
        if not input_tokens:
            return "抱歉，无法理解输入。"
        
        # 转换为张量
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        
        # 生成
        with torch.no_grad():
            if use_kv_cache and hasattr(self.model, 'generate'):
                # 使用模型内置的高效生成方法
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_kv_cache=True
                )
            else:
                # 传统生成方法
                output_ids = self._generate_traditional(
                    input_ids, max_new_tokens, temperature, top_k, top_p, show_progress
                )
        
        # 解码输出（只返回新生成的部分）
        generated_tokens = output_ids[0, len(input_tokens):].tolist()
        return self.decode_tokens(generated_tokens)
    
    def _generate_traditional(self, input_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float, top_k: int, top_p: float, 
                            show_progress: bool = False) -> torch.Tensor:
        """传统生成方法（不使用KV缓存）"""
        generated_ids = input_ids.clone()
        context_window = self.config['model']['context_window']
        
        for step in range(max_new_tokens):
            # 获取当前上下文
            current_context = generated_ids[:, -context_window:]
            
            # 前向传播
            logits = self.model(current_context)
            next_token_logits = logits[:, -1, :]
            
            # 采样下一个token
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # 添加到序列
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 显示进度
            if show_progress and step % 10 == 0:
                print(f"生成进度: {step}/{max_new_tokens}")
        
        return generated_ids
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def interactive_chat(self):
        """交互式聊天"""
        print("\n" + "="*60)
        print("🤖 通用对话小助手已启动！")
        print("💡 您可以和我聊天，我会用西游记的风格回复您")
        print("📝 输入 'quit'、'exit' 或 'bye' 退出")
        print("🎛️ 输入 'settings' 查看生成参数")
        print("="*60)
        
        # 默认生成参数
        settings = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'use_kv_cache': True
        }
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 再见！感谢使用通用对话小助手！")
                    break
                
                if user_input.lower() == 'settings':
                    print("\n⚙️ 当前生成参数:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    continue
                
                # 构建上下文
                context_parts = []
                for item in conversation_history[-3:]:  # 保留最近3轮对话
                    context_parts.append(f"用户：{item['user']}")
                    context_parts.append(f"助手：{item['bot']}")
                
                context_parts.append(f"用户：{user_input}")
                context_parts.append("助手：")
                context_prompt = ''.join(context_parts)
                
                # 生成回复
                print("🤔 思考中...", end="", flush=True)
                start_time = time.time()
                
                response = self.generate(
                    context_prompt,
                    max_new_tokens=settings['max_new_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    use_kv_cache=settings['use_kv_cache']
                )
                
                generation_time = time.time() - start_time
                
                # 清除"思考中..."并显示回复
                print(f"\r{'':20}\r", end="")
                print(f"🤖 助手: {response}")
                print(f"⏱️  生成时间: {generation_time:.2f}秒")
                
                # 添加到历史
                conversation_history.append({
                    'user': user_input,
                    'bot': response
                })
                
                # 保持历史记录不超过10轮
                if len(conversation_history) > 10:
                    conversation_history.pop(0)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！感谢使用通用对话小助手！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("🔄 请重试...")


def warmup_model(model, device, warmup_steps=3):
    """预热模型以获得稳定的推理性能"""
    print(f"🔥 预热模型 ({warmup_steps} 步)...")
    
    model.eval()
    dummy_input = torch.randint(0, 1000, (1, 64), device=device)
    
    with torch.no_grad():
        for i in range(warmup_steps):
            _ = model(dummy_input)
            print(f"  预热步骤 {i+1}/{warmup_steps}")
    
    print("✅ 模型预热完成")


def load_model_and_config(model_dir="model_save_3b", config_path="config/config.yaml", 
                         enable_warmup=True):
    """加载模型和配置的便捷函数"""
    inference = LLMInference3B(model_dir, config_path)
    
    if enable_warmup:
        warmup_model(inference.model, inference.device)
    
    return inference.model, inference.config


def main():
    """主函数"""
    print("🚀 启动3B模型推理...")
    
    try:
        inference = LLMInference3B()
        
        # 测试生成
        test_prompt = "话说天下大势，分久必合，合久必分"
        print(f"\n🧪 测试生成:")
        print(f"输入: {test_prompt}")
        
        result = inference.generate(
            test_prompt,
            max_new_tokens=100,
            temperature=0.8,
            show_progress=True
        )
        print(f"输出: {result}")
        
        # 启动交互式聊天
        inference.interactive_chat()
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        raise


if __name__ == "__main__":
    main()
