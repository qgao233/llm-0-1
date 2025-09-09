#!/usr/bin/env python3
"""
交互式聊天模式 - 与训练好的LLM模型对话
"""

from llmcore import decode_text
from train import download_and_prepare_data
from config_manager import get_config_manager
from infer import sample_next_token, load_model_and_config, warmup_model
import torch
import json
import os
import sys
from typing import List, Dict

class ChatBot:
    def __init__(self, model_dir="./model_save"):
        """初始化聊天机器人"""
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.itos = None
        self.stoi = None
        self.conversation_history = []
        
        # 加载模型和配置
        self._load_model()
        
    def _load_model(self):
        """加载模型和相关配置"""
        try:
            # 询问是否需要预热（加速首次推理）
            warmup_choice = input("🔥 是否预热模型以加速首次推理？(y/n, 默认y): ").strip().lower()
            enable_warmup = warmup_choice != 'n'
            
            # 使用统一的模型加载函数
            self.model, self.config = load_model_and_config(self.model_dir, enable_warmup=enable_warmup)

            # 准备词表
            print("📚 准备词表...")
            dataset, vocab, itos, stoi = download_and_prepare_data()
            self.itos = itos
            self.stoi = stoi
            print("✅ 词表准备完成")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def _encode_text(self, text: str) -> List[int]:
        """将文本编码为token序列"""
        return [self.stoi.get(ch, 0) for ch in text if ch in self.stoi]

    def _decode_tokens(self, tokens: List[int]) -> str:
        """将token序列解码为文本"""
        return ''.join([self.itos.get(idx, '') for idx in tokens])

    def generate_response(self, prompt: str, max_tokens: int = 150, 
                         temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, 
                         show_progress: bool = False) -> str:
        """生成回复"""
        device = self.config['device']
        
        # 编码输入
        input_tokens = self._encode_text(prompt)
        if not input_tokens:
            return "抱歉，我无法理解您的输入。"
            
        # 转换为张量
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
        
        # 生成回复
        generated_tokens = []
        
        if show_progress:
            print("🤖 助手: ", end="", flush=True)
        
        with torch.no_grad():
            for step in range(max_tokens):
                # 获取当前上下文
                context = idx[:, -self.config['context_window']:]
                
                # 前向传播
                logits = self.model(context)
                last_logits = logits[:, -1, :]
                
                # 采样下一个token
                next_token = sample_next_token(
                    last_logits, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p
                )
                
                # 添加到序列
                idx = torch.cat([idx, next_token], dim=-1)
                generated_tokens.append(next_token[0].item())
                
                # 实时显示生成的字符
                if show_progress:
                    new_char = self.itos.get(next_token[0].item(), '')
                    if new_char:  # 只显示有效字符
                        print(new_char, end="", flush=True)
                
                # 检查是否生成了结束符或标点
                new_char = self.itos.get(next_token[0].item(), '')
                
                # 如果生成了换行符且长度足够，也可以停止
                if new_char == '\n' and len(generated_tokens) >= max_tokens - 10:
                    break
        
        if show_progress:
            print()  # 换行
        
        # 解码生成的文本
        response = self._decode_tokens(generated_tokens)
        return response.strip()

    def add_to_history(self, user_input: str, bot_response: str):
        """添加对话到历史记录"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response
        })
        
        # 保持历史记录不超过10轮对话
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def get_context_prompt(self, current_input: str) -> str:
        """构建包含历史对话的提示"""
        context_parts = []
        
        # 添加最近几轮对话作为上下文
        for item in self.conversation_history[-3:]:  # 最多3轮历史
            context_parts.append(f"用户：{item['user']}")
            context_parts.append(f"助手：{item['bot']}")
        
        # 添加当前输入
        context_parts.append(f"用户：{current_input}")
        context_parts.append("助手：")
        
        return ''.join(context_parts)

    def chat(self):
        """开始聊天循环"""
        print("\n" + "="*60)
        print("🤖 西游记AI助手已启动！")
        print("💡 您可以和我聊天，我会用西游记的风格回复您")
        print("📝 输入 'quit'、'exit' 或 'bye' 退出")
        print("🎛️ 输入 'settings' 查看和修改生成参数")
        print("📜 输入 'history' 查看对话历史")
        print("🔄 输入 'clear' 清空对话历史")
        print("⚡ 输入 'stream' 切换实时生成显示模式")
        print("="*60)
        
        # 默认生成参数 - 增加输出长度
        settings = {
            'max_tokens': 150,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9
        }
        
        # 实时生成显示模式
        stream_mode = True  # 默认开启实时显示
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                    
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 再见！感谢使用西游记AI助手！")
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
                    print(f"⚡ 实时生成显示已{status}")
                    continue
                
                # 构建上下文提示
                context_prompt = self.get_context_prompt(user_input)
                
                # 生成回复 - 根据模式决定是否实时显示
                if not stream_mode:
                    print("🤔 思考中...", end="", flush=True)
                
                response = self.generate_response(
                    context_prompt,
                    max_tokens=settings['max_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    show_progress=stream_mode  # 根据用户设置决定是否实时显示
                )
                
                if not stream_mode:
                    print("\r" + " "*20 + "\r", end="")  # 清除"思考中..."
                    print(f"🤖 助手: {response}")
                
                if response:
                    self.add_to_history(user_input, response)
                else:
                    print("🤖 助手: 抱歉，我现在无法回答这个问题。")
                    
            except KeyboardInterrupt:
                print("\n\n👋 再见！感谢使用西游记AI助手！")
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
        print("  max_tokens: 最大生成长度 (20-300)")
        print("  temperature: 随机性 (0.1-2.0, 越小越保守)")
        print("  top_k: 候选词数量 (1-100, 0表示不限制)")
        print("  top_p: 累积概率阈值 (0.1-1.0)")
        
        modify = input("\n🔧 是否要修改参数？(y/n): ").strip().lower()
        if modify == 'y':
            try:
                for key in settings.keys():
                    new_value = input(f"  {key} (当前: {settings[key]}): ").strip()
                    if new_value:
                        if key == 'max_tokens':
                            settings[key] = max(20, min(300, int(new_value)))
                        elif key == 'temperature':
                            settings[key] = max(0.1, min(2.0, float(new_value)))
                        elif key == 'top_k':
                            settings[key] = max(0, min(100, int(new_value)))
                        elif key == 'top_p':
                            settings[key] = max(0.1, min(1.0, float(new_value)))
                print("✅ 参数已更新")
            except ValueError:
                print("❌ 参数格式错误，保持原设置")

    def _show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("📝 暂无对话历史")
            return
            
        print(f"\n📜 对话历史 (最近{len(self.conversation_history)}轮):")
        print("-" * 50)
        for i, item in enumerate(self.conversation_history, 1):
            print(f"{i}. 👤 用户: {item['user']}")
            print(f"   🤖 助手: {item['bot']}")
            print()

def main():
    """主函数"""
    print("🚀 启动西游记AI聊天助手...")
    
    try:
        chatbot = ChatBot()
        chatbot.chat()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已经训练了模型并保存在 ./model_save/ 目录下")

if __name__ == "__main__":
    main()
