#!/usr/bin/env python3
"""
0.1B参数LLM模型推理脚本
支持高效推理、KV缓存、批量生成等功能
专为消费级硬件优化
"""

import torch
import torch.nn.functional as F
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# 添加父目录到路径，确保能导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmcore import create_model
from data_collectors import DataProcessor
from config_manager import InferenceConfig
from checkpoint_manager import InferenceCheckpointManager


def get_device(force_cpu: bool = False, device_config: Dict = None) -> Tuple[torch.device, bool, List[int]]:
    """获取推理设备"""
    if force_cpu:
        return torch.device('cpu'), False, []
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"🎯 检测到 {device_count} 个GPU")
            return torch.device('cuda:0'), True, list(range(device_count))
        else:
            return torch.device('cuda:0'), False, [0]
    else:
        print("⚠️ CUDA不可用，使用CPU")
        return torch.device('cpu'), False, []


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, 
                     top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    """采样下一个token - 0.1B模型优化版本"""
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


class LLMInference:
    """0.1B模型推理器"""
    
    def __init__(self, model_dir: Optional[str] = None, 
                 config_path: str = "pretrain/config/config.yaml"):
        self.config_path = config_path
        self.model = None
        self.config = None
        self.data_processor = None
        self.device = None
        self.itos = None
        self.stoi = None
        
        # 加载配置
        self._load_config()
        
        # 设置模型目录（从配置文件或参数获取）
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            # 优先从推理配置获取模型目录
            inference_config = self.config.get('inference', {})
            if 'model_dir' in inference_config:
                self.model_dir = Path(inference_config['model_dir'])
            else:
                # 回退到训练检查点目录
                checkpoint_dir = self.config.get('checkpoint', {}).get('save_dir', 'checkpoints/0_1b')
                self.model_dir = Path(checkpoint_dir)
        
        print(f"📁 模型目录: {self.model_dir}")
        
        # 设置设备
        self._setup_device()
        
        # 加载模型
        self._load_model()
        
        # 准备分词器
        self._setup_tokenizer()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            # 使用配置管理器加载和验证配置
            config_manager = InferenceConfig(self.config_path)
            self.config = config_manager.config
            print(f"✅ 配置文件加载成功: {self.config_path}")
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _setup_device(self):
        """设置推理设备"""
        device_config = self.config.get('hardware', {})
        self.device, is_multi_gpu, available_gpus = get_device(
            force_cpu=device_config.get('device') == 'cpu'
        )
        print(f"🎯 推理设备: {self.device}")
    
    def _load_model(self):
        """加载训练好的模型"""
        print("🔄 加载0.1B模型...")
        
        # 使用检查点管理器查找模型文件
        checkpoint_manager = InferenceCheckpointManager(str(self.model_dir))
        best_model_path = checkpoint_manager.find_best_model()
        
        if not best_model_path:
            raise FileNotFoundError(f"在 {self.model_dir} 中未找到模型文件")
        
        # 加载检查点
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            
            # 获取模型配置
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
            else:
                model_config = self.config['model']
            
            # 创建模型
            self.model = create_model(model_config)
            
            # 使用检查点管理器加载权重
            checkpoint_manager.load_checkpoint(best_model_path, self.model, strict=False)
            
            # 移动到设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 模型加载成功")
            if 'epoch' in checkpoint:
                print(f"📊 训练轮次: {checkpoint['epoch']}")
            if 'step' in checkpoint:
                print(f"📊 训练步数: {checkpoint['step']}")
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
        vocab_path = data_config.get('vocab_cache_path', 'chinese_tokenizer_vocab_25k.json')
        
        self.data_processor = DataProcessor(
            tokenizer_type=tokenizer_type,
            vocab_path=vocab_path,
            config_path=self.config_path
        )
        
        # 准备词汇表
        try:
            # 尝试从现有的词汇表文件加载
            if os.path.exists(vocab_path):
                print(f"✅ 分词器准备完成，词汇表大小: {self.data_processor.vocab_size or 'Unknown'}")
            else:
                print("⚠️ 词汇表文件不存在，可能需要先训练模型")
            
            # 获取词汇表映射
            if hasattr(self.data_processor, 'tokenizer') and self.data_processor.tokenizer:
                if hasattr(self.data_processor.tokenizer, 'itos'):
                    self.itos = self.data_processor.tokenizer.itos
                if hasattr(self.data_processor.tokenizer, 'stoi'):
                    self.stoi = self.data_processor.tokenizer.stoi
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
    
    def generate(self, prompt: str, max_new_tokens: int = 256, 
                temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
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
        print("🤖 通用对话小助手已启动！(0.1B模型)")
        print("💡 您可以和我聊天，我会回复您")
        print("📝 输入 'quit'、'exit' 或 'bye' 退出")
        print("🎛️ 输入 'settings' 查看生成参数")
        print("="*60)
        
        # 默认生成参数 - 0.1B模型优化
        settings = {
            'max_new_tokens': 256,  # 减少生成长度以适配小模型
            'temperature': 0.8,
            'top_k': 40,           # 减少候选词数量
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
                
                # 构建上下文（保持较短的历史以适配小模型）
                context_parts = []
                for item in conversation_history[-2:]:  # 保留最近2轮对话
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
                
                # 保持历史记录不超过5轮（小模型内存限制）
                if len(conversation_history) > 5:
                    conversation_history.pop(0)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！感谢使用通用对话小助手！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("🔄 请重试...")


def warmup_model(model, device, warmup_steps=2):
    """预热模型以获得稳定的推理性能"""
    print(f"🔥 预热模型 ({warmup_steps} 步)...")
    
    model.eval()
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)  # 减少输入大小
    
    with torch.no_grad():
        for i in range(warmup_steps):
            _ = model(dummy_input)
            print(f"  预热步骤 {i+1}/{warmup_steps}")
    
    print("✅ 模型预热完成")


def load_model_and_config(model_dir=None, 
                         config_path="pretrain/config/config.yaml", 
                         enable_warmup=True):
    """加载模型和配置的便捷函数"""
    inference = LLMInference(model_dir, config_path)
    
    if enable_warmup:
        warmup_model(inference.model, inference.device)
    
    return inference.model, inference.config


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='0.1B LLM模型推理')
    parser.add_argument('--model-dir', default=None, help='模型保存目录（不指定时从配置文件读取）')
    parser.add_argument('--config', default='pretrain/config/config.yaml', help='配置文件路径')
    parser.add_argument('--prompt', default='你好，我想了解', help='测试提示文本')
    parser.add_argument('--max-tokens', type=int, default=50, help='最大生成token数')
    parser.add_argument('--no-chat', action='store_true', help='不启动交互式聊天')
    
    args = parser.parse_args()
    
    print("🚀 启动0.1B模型推理...")
    if args.model_dir:
        print(f"📁 模型目录: {args.model_dir}")
    else:
        print("📁 模型目录: 从配置文件读取")
    print(f"⚙️ 配置文件: {args.config}")
    
    try:
        inference = LLMInference(model_dir=args.model_dir, config_path=args.config)
        
        # 测试生成
        print(f"\n🧪 测试生成:")
        print(f"输入: {args.prompt}")
        
        result = inference.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.8,
            show_progress=True
        )
        print(f"输出: {result}")
        
        # 启动交互式聊天（除非指定不启动）
        if not args.no_chat:
            inference.interactive_chat()
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("💡 请确保已经训练了0.1B模型并保存在相应目录下")
        print("💡 或者检查配置文件中的checkpoint.save_dir设置")
        raise


if __name__ == "__main__":
    main()
