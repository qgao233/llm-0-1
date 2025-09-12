#!/usr/bin/env python3
"""
数据收集器模块 - 支持多种数据类型的收集和处理
为3B模型提供灵活的数据收集策略
整合了数据处理器的功能
"""

import torch
import json
import random
import numpy as np
import os
import urllib.request
import gc
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Iterator
from pathlib import Path
import logging

# 导入中文分词器
try:
    from chinese_tokenizer import ChineseTokenizer, create_chinese_tokenizer
except ImportError:
    print("⚠️ 无法导入中文分词器，将使用字符级分词")
    ChineseTokenizer = None
    create_chinese_tokenizer = None

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理器 - 支持字符级、BPE分词和指令微调数据
    整合到数据收集器系统中
    """
    
    def __init__(self, tokenizer_type: str = "chinese_bpe", vocab_path: Optional[str] = None, 
                 config_path: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            tokenizer_type: 分词器类型 ("char_level" 或 "chinese_bpe")
            vocab_path: 词汇表路径
            config_path: 配置文件路径
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.config = None
        
        # 加载配置文件
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # 初始化分词器
        if tokenizer_type == "chinese_bpe" and create_chinese_tokenizer:
            logger.info("🚀 使用中文优化BPE分词器")
            self.tokenizer = create_chinese_tokenizer()
            if vocab_path:
                # 尝试加载词汇表，如果不存在也不会报错
                self.tokenizer.load_vocab(vocab_path)
        else:
            logger.info("📝 使用字符级分词器")
            self.tokenizer = None
        
        # 数据缓存
        self.dataset = None
        self.vocab = None
        self.itos = None
        self.stoi = None
        self.vocab_size = None
        
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✅ 已加载配置文件: {config_path}")
            
            # 从配置文件更新相关参数
            if self.config and 'data' in self.config:
                data_config = self.config['data']
                
                # 注意：vocab_size 不再从配置文件读取，由分词器自动确定
                
                # 更新词汇表缓存路径
                if 'vocab_cache_path' in data_config:
                    self.vocab_path = data_config['vocab_cache_path']
                
                # 更新分词器类型
                if 'use_bpe_tokenizer' in data_config:
                    self.tokenizer_type = "chinese_bpe" if data_config['use_bpe_tokenizer'] else "char_level"
                    
        except Exception as e:
            logger.warning(f"⚠️ 配置文件加载失败: {e}")
            self.config = None
    
    def encode_text(self, text: str) -> List[int]:
        """编码文本为数字序列"""
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # 字符级编码
            if not self.stoi:
                # 如果没有字符映射，创建一个简单的
                chars = sorted(list(set(text)))
                self.stoi = {ch: i for i, ch in enumerate(chars)}
                self.itos = {i: ch for i, ch in enumerate(chars)}
            return [self.stoi.get(ch, 0) for ch in text]
    
    def decode_text(self, indices: List[int]) -> str:
        """解码数字序列为文本"""
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            # 直接使用BPE分词器解码
            try:
                return self.tokenizer.decode(indices)
            except Exception as e:
                logger.warning(f"⚠️ BPE解码失败: {e}")
                # 降级到逐个解码
                result = []
                for idx in indices:
                    if self.itos and idx in self.itos:
                        result.append(self.itos[idx])
                    else:
                        result.append(f'<UNK_{idx}>')
                return ''.join(result)
        else:
            # 字符级解码
            if not self.itos:
                return ''.join([f'<{idx}>' for idx in indices])
            return ''.join([self.itos.get(idx, '') for idx in indices])
    
    def prepare_vocab_from_data(self, data_sources: List[str]) -> None:
        """从数据源准备词汇表"""
        logger.info("📚 从数据源准备词汇表...")
        
        # 收集所有文本用于训练分词器
        all_texts = []
        for source in data_sources:
            if not Path(source).exists():
                continue
                
            try:
                if source.endswith('.jsonl'):
                    with open(source, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                text = self._format_instruction_item(item)
                                all_texts.append(text)
                elif source.endswith('.json'):
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = self._format_instruction_item(item)
                                all_texts.append(text)
                else:
                    with open(source, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            all_texts.append(text)
            except Exception as e:
                logger.warning(f"⚠️ 处理数据源失败 {source}: {e}")
        
        if not all_texts:
            logger.warning("⚠️ 没有找到有效的文本数据")
            return
        
        # 训练分词器或创建字符级词汇表
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            # 训练BPE分词器（实际上是统计更新，因为基于tiktoken）
            combined_text = '\n'.join(all_texts[:10000])  # 限制训练数据量
            self.tokenizer.train_on_text(combined_text)
            # 获取分词器确定的词汇表大小
            self.vocab_size = self.tokenizer.vocab_size
            self.vocab = list(range(self.vocab_size))
            
            # 保存分词器
            if self.vocab_path:
                self.tokenizer.save_vocab(self.vocab_path)
                logger.info(f"💾 分词器已保存到: {self.vocab_path}")
            else:
                logger.warning("⚠️ vocab_path 为空，跳过保存分词器")
        else:
            # 创建字符级词汇表
            all_chars = set()
            for text in all_texts:
                all_chars.update(text)
            
            self.vocab = sorted(list(all_chars))
            self.itos = {i: ch for i, ch in enumerate(self.vocab)}
            self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
        
        logger.info(f"✅ 词汇表准备完成，大小: {self.vocab_size:,}")
    
    def _format_instruction_item(self, item: Dict) -> str:
        """格式化指令数据项"""
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output_text = item.get('output', '').strip()
        
        if input_text:
            return f"指令：{instruction}\n输入：{input_text}\n回答：{output_text}"
        else:
            return f"指令：{instruction}\n回答：{output_text}"


class BaseDataCollector(ABC):
    """
    基础数据收集器抽象类
    定义所有数据收集器的统一接口
    """
    
    def __init__(self, name: str, data_processor, config: Dict = None):
        """
        初始化数据收集器
        
        Args:
            name: 收集器名称
            data_processor: 数据处理器实例
            config: 配置参数
        """
        self.name = name
        self.data_processor = data_processor
        self.config = config or {}
        self.data_cache = []
        self.current_index = 0
        
    @abstractmethod
    def load_data(self, data_source: Union[str, List[str]]) -> None:
        """
        加载数据源
        
        Args:
            data_source: 数据源路径或路径列表
        """
        pass
    
    @abstractmethod
    def get_sample(self, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练样本
        
        Args:
            seq_length: 序列长度
            
        Returns:
            (input_tensor, target_tensor): 输入和目标张量
        """
        pass
    
    @abstractmethod
    def get_batch(self, batch_size: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个批次的训练样本
        
        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            
        Returns:
            (input_batch, target_batch): 输入和目标批次张量
        """
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_cache)
    
    def reset(self) -> None:
        """重置数据索引"""
        self.current_index = 0
        if self.config.get('shuffle', True):
            random.shuffle(self.data_cache)


class ConversationDataCollector(BaseDataCollector):
    """
    对话数据收集器
    处理instruction-output格式的对话数据
    """
    
    def __init__(self, data_processor, config: Dict = None):
        super().__init__("ConversationCollector", data_processor, config)
        self.conversation_templates = [
            "指令：{instruction}\n回答：{output}",
            "问题：{instruction}\n答案：{output}",
            "用户：{instruction}\n助手：{output}",
            "Human: {instruction}\nAssistant: {output}",
        ]
        self.current_template = 0
        
    def load_data(self, data_source: Union[str, List[str]]) -> None:
        """加载对话数据"""
        logger.info(f"🗣️ {self.name} 加载对话数据...")
        
        if isinstance(data_source, str):
            data_sources = [data_source]
        else:
            data_sources = data_source
        
        self.data_cache = []
        
        for source in data_sources:
            if not Path(source).exists():
                logger.warning(f"⚠️ 数据文件不存在: {source}")
                continue
                
            try:
                # 支持JSON和JSONL格式
                if source.endswith('.jsonl'):
                    with open(source, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                self.data_cache.append(item)
                else:
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.data_cache.extend(data)
                        else:
                            self.data_cache.append(data)
                            
                logger.info(f"  ✅ 从 {source} 加载了 {len(self.data_cache)} 个对话样本")
                
            except Exception as e:
                logger.error(f"  ❌ 加载数据失败 {source}: {e}")
        
        logger.info(f"📊 {self.name} 总共加载了 {len(self.data_cache)} 个对话样本")
        
        # 打乱数据
        if self.config.get('shuffle', True):
            random.shuffle(self.data_cache)
    
    def _format_conversation(self, item: Dict) -> str:
        """格式化对话数据"""
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output_text = item.get('output', '').strip()
        
        # 选择模板
        template = self.conversation_templates[self.current_template % len(self.conversation_templates)]
        
        # 处理有输入的情况
        if input_text:
            full_instruction = f"{instruction}\n输入：{input_text}"
        else:
            full_instruction = instruction
        
        # 格式化对话
        formatted = template.format(instruction=full_instruction, output=output_text)
        
        # 轮换模板以增加多样性
        self.current_template += 1
        
        return formatted
    
    def get_sample(self, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个对话样本"""
        if not self.data_cache:
            raise ValueError("数据缓存为空，请先调用load_data()")
        
        # 循环获取样本
        if self.current_index >= len(self.data_cache):
            self.reset()
        
        # 获取对话数据
        conversation_item = self.data_cache[self.current_index]
        self.current_index += 1
        
        # 格式化对话
        formatted_text = self._format_conversation(conversation_item)
        
        # 编码文本
        tokens = self.data_processor.encode_text(formatted_text)
        
        # 如果tokens太短，尝试合并多个对话
        while len(tokens) < seq_length and self.current_index < len(self.data_cache):
            next_item = self.data_cache[self.current_index]
            next_text = self._format_conversation(next_item)
            next_tokens = self.data_processor.encode_text(next_text)
            
            # 添加分隔符
            separator_tokens = self.data_processor.encode_text("\n\n")
            tokens.extend(separator_tokens + next_tokens)
            self.current_index += 1
        
        # 截断或填充到指定长度
        if len(tokens) > seq_length + 1:
            tokens = tokens[:seq_length + 1]
        elif len(tokens) < seq_length + 1:
            # 用pad token填充（通常是0）
            pad_token = 0
            tokens.extend([pad_token] * (seq_length + 1 - len(tokens)))
        
        # 创建输入和目标张量
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y
    
    def get_batch(self, batch_size: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个批次的对话样本"""
        batch_x = []
        batch_y = []
        
        for _ in range(batch_size):
            x, y = self.get_sample(seq_length)
            batch_x.append(x)
            batch_y.append(y)
        
        return torch.stack(batch_x), torch.stack(batch_y)


class LongTextDataCollector(BaseDataCollector):
    """
    长文本序列数据收集器
    处理连续的长文本数据，适合预训练
    """
    
    def __init__(self, data_processor, config: Dict = None):
        super().__init__("LongTextCollector", data_processor, config)
        self.text_data = ""
        self.tokens = []
        
    def load_data(self, data_source: Union[str, List[str]]) -> None:
        """加载长文本数据"""
        logger.info(f"📖 {self.name} 加载长文本数据...")
        
        if isinstance(data_source, str):
            data_sources = [data_source]
        else:
            data_sources = data_source
        
        all_texts = []
        
        for source in data_sources:
            if not Path(source).exists():
                logger.warning(f"⚠️ 数据文件不存在: {source}")
                continue
                
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        all_texts.append(text)
                        logger.info(f"  ✅ 从 {source} 加载了 {len(text):,} 个字符")
                        
            except Exception as e:
                logger.error(f"  ❌ 加载数据失败 {source}: {e}")
        
        # 合并所有文本
        self.text_data = '\n\n'.join(all_texts)
        
        # 编码整个文本
        self.tokens = self.data_processor.encode_text(self.text_data)
        
        logger.info(f"📊 {self.name} 总共加载了 {len(self.text_data):,} 个字符，{len(self.tokens):,} 个tokens")
        
        # 创建数据缓存（滑动窗口索引）
        self.data_cache = list(range(len(self.tokens)))
        
        if self.config.get('shuffle', True):
            random.shuffle(self.data_cache)
    
    def get_sample(self, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个长文本样本"""
        if not self.tokens:
            raise ValueError("tokens为空，请先调用load_data()")
        
        if len(self.tokens) <= seq_length:
            raise ValueError(f"文本长度 {len(self.tokens)} 必须大于序列长度 {seq_length}")
        
        # 随机选择起始位置
        max_start = len(self.tokens) - seq_length - 1
        start_idx = random.randint(0, max_start)
        
        # 获取序列
        x = torch.tensor(self.tokens[start_idx:start_idx + seq_length], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + seq_length + 1], dtype=torch.long)
        
        return x, y
    
    def get_batch(self, batch_size: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个批次的长文本样本"""
        batch_x = []
        batch_y = []
        
        for _ in range(batch_size):
            x, y = self.get_sample(seq_length)
            batch_x.append(x)
            batch_y.append(y)
        
        return torch.stack(batch_x), torch.stack(batch_y)


class CodeDataCollector(BaseDataCollector):
    """
    代码数据收集器
    专门处理代码训练数据
    """
    
    def __init__(self, data_processor, config: Dict = None):
        super().__init__("CodeCollector", data_processor, config)
        self.code_templates = [
            "# 代码示例\n{code}",
            "```python\n{code}\n```",
            "以下是Python代码：\n{code}",
        ]
        
    def load_data(self, data_source: Union[str, List[str]]) -> None:
        """加载代码数据"""
        logger.info(f"💻 {self.name} 加载代码数据...")
        
        if isinstance(data_source, str):
            data_sources = [data_source]
        else:
            data_sources = data_source
        
        self.data_cache = []
        
        for source in data_sources:
            if not Path(source).exists():
                logger.warning(f"⚠️ 数据文件不存在: {source}")
                continue
                
            try:
                # 支持.py文件和JSON格式
                if source.endswith('.py'):
                    with open(source, 'r', encoding='utf-8') as f:
                        code_content = f.read().strip()
                        if code_content:
                            self.data_cache.append({'code': code_content})
                else:
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.data_cache.extend(data)
                        else:
                            self.data_cache.append(data)
                            
                logger.info(f"  ✅ 从 {source} 加载了代码数据")
                
            except Exception as e:
                logger.error(f"  ❌ 加载数据失败 {source}: {e}")
        
        logger.info(f"📊 {self.name} 总共加载了 {len(self.data_cache)} 个代码样本")
        
        if self.config.get('shuffle', True):
            random.shuffle(self.data_cache)
    
    def get_sample(self, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个代码样本"""
        if not self.data_cache:
            raise ValueError("数据缓存为空，请先调用load_data()")
        
        if self.current_index >= len(self.data_cache):
            self.reset()
        
        # 获取代码数据
        code_item = self.data_cache[self.current_index]
        self.current_index += 1
        
        # 格式化代码
        code_content = code_item.get('code', '')
        template = random.choice(self.code_templates)
        formatted_text = template.format(code=code_content)
        
        # 编码文本
        tokens = self.data_processor.encode_text(formatted_text)
        
        # 处理长度
        if len(tokens) > seq_length + 1:
            tokens = tokens[:seq_length + 1]
        elif len(tokens) < seq_length + 1:
            pad_token = 0
            tokens.extend([pad_token] * (seq_length + 1 - len(tokens)))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y
    
    def get_batch(self, batch_size: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个批次的代码样本"""
        batch_x = []
        batch_y = []
        
        for _ in range(batch_size):
            x, y = self.get_sample(seq_length)
            batch_x.append(x)
            batch_y.append(y)
        
        return torch.stack(batch_x), torch.stack(batch_y)


class MixedDataset(torch.utils.data.Dataset):
    """
    混合数据集类
    随机调用不同的数据收集器获取训练样本
    """
    
    def __init__(self, collectors: List[BaseDataCollector], seq_length: int, 
                 weights: Optional[List[float]] = None, config: Dict = None):
        """
        初始化混合数据集
        
        Args:
            collectors: 数据收集器列表
            seq_length: 序列长度
            weights: 各收集器的采样权重
            config: 配置参数
        """
        self.collectors = collectors
        self.seq_length = seq_length
        self.config = config or {}
        
        # 设置采样权重
        if weights is None:
            self.weights = [1.0] * len(collectors)
        else:
            if len(weights) != len(collectors):
                raise ValueError("权重数量必须与收集器数量相同")
            self.weights = weights
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 计算总数据集大小（取最大的收集器大小）
        self.total_size = max(len(collector) for collector in collectors) if collectors else 0
        
        logger.info(f"🔀 混合数据集初始化完成:")
        for i, collector in enumerate(collectors):
            logger.info(f"  📊 {collector.name}: {len(collector):,} 样本, 权重: {self.weights[i]:.3f}")
        logger.info(f"  📈 总数据集大小: {self.total_size:,}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个训练样本"""
        # 根据权重随机选择收集器
        collector = np.random.choice(self.collectors, p=self.weights)
        
        # 从选中的收集器获取样本
        try:
            x, y = collector.get_sample(self.seq_length)
            return x, y
        except Exception as e:
            logger.warning(f"⚠️ 从 {collector.name} 获取样本失败: {e}")
            # 降级到第一个收集器
            return self.collectors[0].get_sample(self.seq_length)
    
    def get_batch_from_collector(self, collector_name: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从指定收集器获取批次"""
        for collector in self.collectors:
            if collector.name == collector_name:
                return collector.get_batch(batch_size, self.seq_length)
        
        raise ValueError(f"未找到收集器: {collector_name}")
    
    def reset_all_collectors(self) -> None:
        """重置所有收集器"""
        for collector in self.collectors:
            collector.reset()
    
    def get_collector_stats(self) -> Dict:
        """获取收集器统计信息"""
        stats = {}
        for i, collector in enumerate(self.collectors):
            stats[collector.name] = {
                'size': len(collector),
                'weight': self.weights[i],
                'current_index': collector.current_index
            }
        return stats


def create_data_collectors(config: Dict) -> Tuple[List[BaseDataCollector], DataProcessor]:
    """
    根据配置创建数据收集器和数据处理器
    
    Args:
        config: 配置字典
        
    Returns:
        (数据收集器列表, 数据处理器)
    """
    # 创建数据处理器
    data_config = config.get('data', {})
    tokenizer_type = "chinese_bpe" if data_config.get('use_bpe_tokenizer', True) else "char_level"
    vocab_path = data_config.get('vocab_cache_path', 'chinese_tokenizer_vocab_25k.json')
    logger.info(f"🔍 调试: 从配置获取的 vocab_path = '{vocab_path}'")
    
    data_processor = DataProcessor(
        tokenizer_type=tokenizer_type,
        vocab_path=vocab_path,
        config_path=None  # 不需要重复加载配置，直接使用传入的config
    )
    
    # 收集所有数据源用于准备词汇表
    all_data_sources = []
    collector_configs = config.get('data_collectors', {})
    
    # 收集对话数据源
    if collector_configs.get('conversation', {}).get('enabled', True):
        conv_sources = collector_configs.get('conversation', {}).get('data_sources', [])
        all_data_sources.extend(conv_sources)
    
    # 收集长文本数据源
    if collector_configs.get('long_text', {}).get('enabled', False):
        text_sources = collector_configs.get('long_text', {}).get('data_sources', [])
        all_data_sources.extend(text_sources)
    
    # 收集代码数据源
    if collector_configs.get('code', {}).get('enabled', False):
        code_sources = collector_configs.get('code', {}).get('data_sources', [])
        all_data_sources.extend(code_sources)
    
    # 如果没有配置数据源，使用默认配置
    if not all_data_sources:
        raise ValueError("未配置任何数据源，请在配置文件中指定数据源")
    
    # 准备词汇表
    if all_data_sources:
        data_processor.prepare_vocab_from_data(all_data_sources)
    
    collectors = []
    
    # 创建对话数据收集器
    if collector_configs.get('conversation', {}).get('enabled', True):
        conv_config = collector_configs.get('conversation', {})
        conv_collector = ConversationDataCollector(data_processor, conv_config)
        
        # 加载对话数据
        conv_data_sources = conv_config.get('data_sources', [])
        if conv_data_sources:
            conv_collector.load_data(conv_data_sources)
            collectors.append(conv_collector)
            logger.info(f"✅ 创建对话数据收集器: {len(conv_collector)} 样本")
    
    # 创建长文本数据收集器
    if collector_configs.get('long_text', {}).get('enabled', False):
        text_config = collector_configs.get('long_text', {})
        text_collector = LongTextDataCollector(data_processor, text_config)
        
        # 加载长文本数据
        text_data_sources = text_config.get('data_sources', [])
        if text_data_sources:
            text_collector.load_data(text_data_sources)
            collectors.append(text_collector)
            logger.info(f"✅ 创建长文本数据收集器: {len(text_collector)} tokens")
    
    # 创建代码数据收集器
    if collector_configs.get('code', {}).get('enabled', False):
        code_config = collector_configs.get('code', {})
        code_collector = CodeDataCollector(data_processor, code_config)
        
        # 加载代码数据
        code_data_sources = code_config.get('data_sources', [])
        if code_data_sources:
            code_collector.load_data(code_data_sources)
            collectors.append(code_collector)
            logger.info(f"✅ 创建代码数据收集器: {len(code_collector)} 样本")
    
    if not collectors:
        logger.warning("⚠️ 未创建任何数据收集器，将使用默认对话收集器")
        # 创建默认对话收集器
        default_collector = ConversationDataCollector(data_processor)
        # 尝试加载默认数据
        if all_data_sources:
            default_collector.load_data(all_data_sources)
        collectors.append(default_collector)
    
    return collectors, data_processor


# 兼容性函数
def create_data_processor_from_config(config_path: str) -> DataProcessor:
    """
    从配置文件创建数据处理器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataProcessor实例
    """
    return DataProcessor(config_path=config_path)


if __name__ == "__main__":
    # 测试数据收集器
    print("🧪 测试数据收集器系统...")
    
    # 创建测试配置
    test_config = {
        'data': {
            'use_bpe_tokenizer': True,
            'vocab_cache_path': 'test_vocab.json',
        },
        'model': {
            'vocab_size': 10000
        },
        'data_collectors': {
            'conversation': {
                'enabled': True,
                'data_sources': ["data/train_2M_CN_split/train_2M_CN_part_01.jsonl"],
                'shuffle': True
            }
        }
    }
    
    # 测试创建数据收集器
    print("\n1. 测试创建数据收集器:")
    try:
        collectors, processor = create_data_collectors(test_config)
        print(f"✅ 创建了 {len(collectors)} 个数据收集器")
        print(f"✅ 数据处理器类型: {processor.tokenizer_type}")
        
        if collectors:
            # 测试获取样本
            conv_collector = collectors[0]
            if len(conv_collector) > 0:
                x, y = conv_collector.get_sample(128)
                print(f"✅ 对话样本形状: x={x.shape}, y={y.shape}")
                
                batch_x, batch_y = conv_collector.get_batch(2, 128)
                print(f"✅ 对话批次形状: x={batch_x.shape}, y={batch_y.shape}")
            else:
                print("⚠️ 收集器中没有数据")
        
    except Exception as e:
        print(f"⚠️ 数据收集器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试混合数据集
    print("\n2. 测试混合数据集:")
    try:
        if 'collectors' in locals() and collectors:
            mixed_dataset = MixedDataset(collectors, seq_length=128)
            
            print(f"✅ 混合数据集大小: {len(mixed_dataset)}")
            
            # 测试获取样本
            if len(mixed_dataset) > 0:
                x, y = mixed_dataset[0]
                print(f"✅ 混合样本形状: x={x.shape}, y={y.shape}")
                
                # 打印统计信息
                stats = mixed_dataset.get_collector_stats()
                print(f"📊 收集器统计: {stats}")
            else:
                print("⚠️ 混合数据集为空")
        else:
            print("⚠️ 没有可用的收集器")
        
    except Exception as e:
        print(f"⚠️ 混合数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 数据收集器系统测试完成！")
