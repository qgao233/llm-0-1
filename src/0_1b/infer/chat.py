#!/usr/bin/env python3
"""
0.1B参数LLM模型交互式聊天脚本
支持流式输出、对话历史、参数调整等功能
专为消费级硬件优化
"""

import torch
import os
import sys
import time
import json
from typing import List, Dict
from pathlib import Path

from infer import LLMInference, sample_next_token


class ChatBot:
    """0.1B模型聊天机器人"""
    
    def __init__(self, model_dir=None, 
                 config_path="pretrain/config/config.yaml"):
        """初始化0.1B聊天机器人"""
        self.model_dir = model_dir
        self.config_path = config_path
        self.conversation_history = []
        
        # 加载模型和配置
        print("🚀 正在加载0.1B模型...")
        self._load_model()
        
    def _load_model(self):
        """加载模型和相关配置"""
        try:
            # 询问是否需要预热
            warmup_choice = input("🔥 是否预热模型以加速推理？(y/n, 默认y): ").strip().lower()
            enable_warmup = warmup_choice != 'n'
            
            # 创建推理器
            self.inference = LLMInference(self.model_dir, self.config_path)
            
            # 预热模型
            if enable_warmup:
                self._warmup_model()
            
            print("✅ 0.1B模型加载完成")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 请确保已经训练了0.1B模型并保存在相应目录下")
            sys.exit(1)
    
    def _warmup_model(self):
        """预热模型"""
        print("🔥 预热0.1B模型中...")
        dummy_prompt = "测试"
        
        start_time = time.time()
        _ = self.inference.generate(
            dummy_prompt,
            max_new_tokens=10,
            temperature=0.8,
            show_progress=False
        )
        warmup_time = time.time() - start_time
        
        print(f"✅ 模型预热完成，用时: {warmup_time:.2f}秒")
    
    def generate_response(self, prompt: str, max_tokens: int = 256, 
                         temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                         use_kv_cache: bool = True, show_progress: bool = False, 
                         stream_delay: float = 0.03) -> str:
        """生成回复"""
        try:
            # 生成完整回复
            response = self.inference.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_kv_cache=use_kv_cache,
                show_progress=False  # 内部不显示进度
            )
            
            # 如果需要流式显示
            if show_progress and response:
                self._fake_stream_output(response, delay=stream_delay)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ 生成回复时发生错误: {e}")
            return "抱歉，我现在无法回答这个问题。"
    
    def _fake_stream_output(self, text: str, delay: float = 0.03):
        """假的流式输出效果"""
        print("🤖 助手: ", end="", flush=True)
        
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
        
        print()  # 换行
    
    def add_to_history(self, user_input: str, bot_response: str):
        """添加对话到历史记录"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': time.time()
        })
        
        # 保持历史记录不超过8轮对话（0.1B模型内存限制）
        if len(self.conversation_history) > 8:
            self.conversation_history.pop(0)
    
    def get_context_prompt(self, current_input: str) -> str:
        """构建包含历史对话的提示"""
        context_parts = []
        
        # 添加最近几轮对话作为上下文（0.1B模型处理较短上下文）
        for item in self.conversation_history[-3:]:  # 最多3轮历史
            context_parts.append(f"用户：{item['user']}")
            context_parts.append(f"助手：{item['bot']}")
        
        # 添加当前输入
        context_parts.append(f"用户：{current_input}")
        context_parts.append("助手：")
        
        return ''.join(context_parts)
    
    def chat(self):
        """开始聊天循环"""
        print("\n" + "="*70)
        print("🤖 通用对话小助手已启动！")
        print("💡 这是一个0.1亿参数的大模型，适合轻量级对话")
        print("📚 您可以和我聊天，我会回复您")
        print("📝 输入 'quit'、'exit' 或 'bye' 退出")
        print("🎛️ 输入 'settings' 查看和修改生成参数")
        print("📜 输入 'history' 查看对话历史")
        print("🔄 输入 'clear' 清空对话历史")
        print("⚡ 输入 'stream' 切换流式输出显示模式")
        print("💾 输入 'save' 保存对话历史")
        print("📊 输入 'stats' 查看模型统计信息")
        print("="*70)
        
        # 默认生成参数 - 0.1B模型优化
        settings = {
            'max_tokens': 256,      # 适中的生成长度
            'temperature': 0.8,     # 适中的随机性
            'top_k': 40,           # 较少的候选词数量
            'top_p': 0.9,
            'use_kv_cache': True,   # 启用KV缓存以提高效率
            'stream_delay': 0.03    # 稍慢的流式输出延迟
        }
        
        # 流式输出模式
        stream_mode = True
        
        # 统计信息
        total_tokens_generated = 0
        total_generation_time = 0
        conversation_count = 0
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self._show_session_stats(conversation_count, total_tokens_generated, total_generation_time)
                    print("👋 再见！感谢使用通用对话小助手！")
                    break
                
                elif user_input.lower() == 'settings':
                    self._show_settings(settings)
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🗑️ 对话历史已清空")
                    continue
                
                elif user_input.lower() == 'stream':
                    stream_mode = not stream_mode
                    status = "开启" if stream_mode else "关闭"
                    print(f"⚡ 流式输出显示已{status}")
                    continue
                
                elif user_input.lower() == 'save':
                    self._save_conversation()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_model_stats()
                    continue
                
                # 构建上下文提示
                context_prompt = self.get_context_prompt(user_input)
                
                # 显示思考中提示
                thinking_text = "🤔 0.1B模型思考中..."
                print(thinking_text, end="", flush=True)
                
                # 生成回复
                start_time = time.time()
                response = self.generate_response(
                    context_prompt,
                    max_tokens=settings['max_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    use_kv_cache=settings['use_kv_cache'],
                    show_progress=stream_mode,
                    stream_delay=settings['stream_delay']
                )
                generation_time = time.time() - start_time
                
                # 清除思考中提示
                clear_spaces = " " * (len(thinking_text) + 5)
                print(f"\r{clear_spaces}\r", end="", flush=True)
                
                # 如果不是流式模式，直接显示完整回复
                if not stream_mode:
                    print(f"🤖 助手: {response}")
                
                # 显示生成统计
                if response:
                    # 估算生成的token数量
                    estimated_tokens = len(response) // 2  # 粗略估算
                    tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
                    
                    print(f"📊 生成统计: {estimated_tokens} tokens, {generation_time:.2f}s, {tokens_per_second:.1f} tokens/s")
                    
                    # 更新统计
                    total_tokens_generated += estimated_tokens
                    total_generation_time += generation_time
                    conversation_count += 1
                    
                    # 添加到历史
                    self.add_to_history(user_input, response)
                else:
                    print("🤖 助手: 抱歉，我现在无法回答这个问题。")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！感谢使用通用对话小助手！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("🔄 请重试...")
    
    def _show_settings(self, settings: Dict):
        """显示和修改设置"""
        print("\n⚙️ 当前生成参数:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\n💡 参数说明:")
        print("  max_tokens: 最大生成长度 (50-512)")
        print("  temperature: 随机性 (0.1-2.0, 越小越保守)")
        print("  top_k: 候选词数量 (1-100, 0表示不限制)")
        print("  top_p: 累积概率阈值 (0.1-1.0)")
        print("  use_kv_cache: 是否使用KV缓存 (true/false)")
        print("  stream_delay: 流式输出延迟秒数 (0.01-0.1)")
        
        modify = input("\n🔧 是否要修改参数？(y/n): ").strip().lower()
        if modify == 'y':
            try:
                for key in settings.keys():
                    if key == 'use_kv_cache':
                        new_value = input(f"  {key} (当前: {settings[key]}): ").strip()
                        if new_value.lower() in ['true', 'false']:
                            settings[key] = new_value.lower() == 'true'
                    else:
                        new_value = input(f"  {key} (当前: {settings[key]}): ").strip()
                        if new_value:
                            if key == 'max_tokens':
                                settings[key] = max(50, min(512, int(new_value)))
                            elif key == 'temperature':
                                settings[key] = max(0.1, min(2.0, float(new_value)))
                            elif key == 'top_k':
                                settings[key] = max(0, min(100, int(new_value)))
                            elif key == 'top_p':
                                settings[key] = max(0.1, min(1.0, float(new_value)))
                            elif key == 'stream_delay':
                                settings[key] = max(0.01, min(0.1, float(new_value)))
                print("✅ 参数已更新")
            except ValueError:
                print("❌ 参数格式错误，保持原设置")
    
    def _show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("📝 暂无对话历史")
            return
        
        print(f"\n📜 对话历史 (最近{len(self.conversation_history)}轮):")
        print("-" * 60)
        for i, item in enumerate(self.conversation_history, 1):
            timestamp = time.strftime("%H:%M:%S", time.localtime(item['timestamp']))
            print(f"{i}. [{timestamp}] 👤 用户: {item['user']}")
            print(f"   🤖 助手: {item['bot']}")
            print()
    
    def _save_conversation(self):
        """保存对话历史"""
        if not self.conversation_history:
            print("📝 暂无对话历史可保存")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_0_1b_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"💾 对话历史已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def _show_model_stats(self):
        """显示模型统计信息"""
        print("\n📊 0.1B模型统计信息:")
        
        # 模型参数信息
        total_params = sum(p.numel() for p in self.inference.model.parameters())
        trainable_params = sum(p.numel() for p in self.inference.model.parameters() if p.requires_grad)
        
        print(f"  🏗️  模型架构:")
        print(f"    - 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"    - 可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"    - 隐藏维度: {self.inference.config['model']['d_model']}")
        print(f"    - 注意力头数: {self.inference.config['model']['n_heads']}")
        print(f"    - 层数: {self.inference.config['model']['n_layers']}")
        print(f"    - 上下文窗口: {self.inference.config['model']['context_window']}")
        print(f"    - 词汇表大小: {self.inference.config['model'].get('vocab_size', 'Dynamic')}")
        
        # 设备信息
        print(f"  🎯 运行环境:")
        print(f"    - 设备: {self.inference.device}")
        if self.inference.device.type == 'cuda':
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_name(0)
                print(f"    - GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 对话统计
        print(f"  💬 对话统计:")
        print(f"    - 历史对话轮数: {len(self.conversation_history)}")
        if self.conversation_history:
            session_time = time.time() - self.conversation_history[0]['timestamp']
            print(f"    - 会话时长: {session_time/60:.1f} 分钟")
    
    def _show_session_stats(self, conversation_count: int, total_tokens: int, total_time: float):
        """显示会话统计"""
        if conversation_count > 0:
            print(f"\n📊 本次会话统计:")
            print(f"  💬 对话轮数: {conversation_count}")
            print(f"  🔤 生成token数: {total_tokens}")
            print(f"  ⏱️  总生成时间: {total_time:.2f}秒")
            print(f"  🚀 平均生成速度: {total_tokens/total_time:.1f} tokens/s")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='0.1B LLM聊天机器人')
    parser.add_argument('--model-dir', default=None, help='模型保存目录（不指定时从配置文件读取）')
    parser.add_argument('--config', default='pretrain/config/config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    print("🚀 启动通用对话小聊天助手...")
    if args.model_dir:
        print(f"📁 模型目录: {args.model_dir}")
    else:
        print("📁 模型目录: 从配置文件读取")
    print(f"⚙️ 配置文件: {args.config}")
    
    try:
        chatbot = ChatBot(args.model_dir, args.config)
        chatbot.chat()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已经训练了0.1B模型并保存在相应目录下")
        print("💡 或者使用不同的参数:")
        print("   python chat.py --model-dir /path/to/model --config pretrain/config/config.yaml")
        print("   python chat.py --config pretrain/config/config.yaml  # 从配置文件读取模型目录")


if __name__ == "__main__":
    main()
