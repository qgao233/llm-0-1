#!/usr/bin/env python3
"""
数据处理模块 - 使用中文优化分词器
整合tiktoken分词器和原有的数据处理流程
"""

import torch
import os
import urllib.request
import numpy as np
from typing import Dict, List, Tuple, Optional
from chinese_tokenizer import ChineseTokenizer, create_chinese_tokenizer

class DataProcessor:
    """
    数据处理器 - 支持字符级和BPE分词
    """
    
    def __init__(self, tokenizer_type: str = "chinese_bpe", vocab_path: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            tokenizer_type: 分词器类型 ("char_level" 或 "chinese_bpe")
            vocab_path: 词汇表路径
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_path = vocab_path
        
        if tokenizer_type == "chinese_bpe":
            print("🚀 使用中文优化BPE分词器")
            self.tokenizer = create_chinese_tokenizer()
            if vocab_path and os.path.exists(vocab_path):
                self.tokenizer.load_vocab(vocab_path)
        else:
            print("📝 使用字符级分词器")
            self.tokenizer = None
        
        # 数据缓存
        self.dataset = None
        self.vocab = None
        self.itos = None
        self.stoi = None
        self.vocab_size = None
        
    def download_and_prepare_data(self, data_file: str = "xiyouji.txt", 
                                force_download: bool = False) -> Tuple[torch.Tensor, List, Dict, Dict]:
        """
        下载并准备数据集
        
        Args:
            data_file: 数据文件名
            force_download: 是否强制重新下载
            
        Returns:
            (dataset, vocab, itos, stoi)
        """
        # 下载数据文件
        if not os.path.exists(data_file) or force_download:
            print(f"📥 正在下载数据集到 {data_file}...")
            url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
            try:
                urllib.request.urlretrieve(url, data_file)
                print("✅ 数据集下载完成")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                # 创建示例数据
                self._create_sample_data(data_file)
        
        # 读取数据
        with open(data_file, 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        print(f"📊 数据文件大小: {len(text_data):,} 字符")
        
        # 根据分词器类型处理数据
        if self.tokenizer_type == "chinese_bpe":
            return self._prepare_bpe_data(text_data)
        else:
            return self._prepare_char_data(text_data)
    
    def _create_sample_data(self, data_file: str):
        """创建示例数据（如果下载失败）"""
        sample_text = """西游记是中国古代神话小说，讲述了孙悟空、猪八戒、沙和尚保护唐僧西天取经的故事。
孙悟空本领高强，能七十二变，一个筋斗云十万八千里。
猪八戒贪吃懒做，但心地善良。沙和尚忠厚老实，任劳任怨。
唐僧心慈面善，一心向佛，但有时过于仁慈。
他们师徒四人历经九九八十一难，终于取得真经，修成正果。
这是一个关于友谊、坚持和成长的故事。
人工智能技术发展迅速，深度学习和机器学习成为热门领域。
自然语言处理让计算机能够理解和生成人类语言。
大模型训练需要大量的数据和计算资源。
分词器是自然语言处理的重要组成部分。"""
        
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"✅ 创建示例数据文件: {data_file}")
    
    def _prepare_bpe_data(self, text_data: str) -> Tuple[torch.Tensor, List, Dict, Dict]:
        """使用BPE分词器准备数据"""
        print("🔨 使用BPE分词器处理数据...")
        
        # 使用分词器编码
        tokens = self.tokenizer.encode(text_data)
        print(f"📊 编码结果: {len(tokens):,} tokens")
        
        # 获取词汇表大小
        self.vocab_size = self.tokenizer.vocab_size
        
        # 创建正确的词汇表映射 - 直接使用分词器的解码功能
        self.vocab = list(range(self.vocab_size))
        
        # 为BPE分词器创建简化的itos映射
        # 注意：对于BPE，我们主要依赖tokenizer.decode()进行解码
        # itos主要用于兼容性和调试
        self.itos = {}
        print("🔨 构建BPE词汇表映射（快速模式）...")
        
        # 只为前1000个token构建映射，用于调试和兼容性
        max_mapping = min(1000, self.vocab_size)
        for token_id in range(max_mapping):
            try:
                decoded = self.tokenizer.decode([token_id])
                self.itos[token_id] = decoded
            except:
                self.itos[token_id] = f'<UNK_{token_id}>'
        
        # 为其余token使用默认映射
        for token_id in range(max_mapping, self.vocab_size):
            self.itos[token_id] = f'<UNK_{token_id}>'
        
        # 创建反向映射（虽然在BPE中不常用）
        self.stoi = {v: k for k, v in self.itos.items() if not v.startswith('<UNK_')}
        
        # 转换为tensor
        self.dataset = torch.tensor(tokens, dtype=torch.long)
        
        print(f"📈 BPE词汇表大小: {self.vocab_size:,}")
        print(f"📊 数据集形状: {self.dataset.shape}")
        print(f"📝 itos映射样例: {list(self.itos.items())[:10]}")
        
        # 打印压缩统计
        self.tokenizer.print_stats()
        
        # 保存词汇表
        if self.vocab_path:
            self.tokenizer.save_vocab(self.vocab_path)
        
        return self.dataset, self.vocab, self.itos, self.stoi
    
    def _prepare_char_data(self, text_data: str) -> Tuple[torch.Tensor, List, Dict, Dict]:
        """使用字符级分词器准备数据（原有方法）"""
        print("🔨 使用字符级分词器处理数据...")
        
        # 创建字符级词表
        self.vocab = sorted(list(set(text_data)))
        print(f"📈 字符级词表大小: {len(self.vocab)}")
        
        # 创建编码映射
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        
        # 编码整个数据集
        self.dataset = torch.tensor([self.stoi[ch] for ch in text_data], dtype=torch.long)
        print(f"📊 数据集形状: {self.dataset.shape}")
        
        self.vocab_size = len(self.vocab)
        
        return self.dataset, self.vocab, self.itos, self.stoi
    
    def encode_text(self, text: str) -> List[int]:
        """编码文本为数字序列"""
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # 字符级编码
            return [self.stoi.get(ch, 0) for ch in text if ch in self.stoi]
    
    def decode_text(self, indices: List[int]) -> str:
        """解码数字序列为文本"""
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            # 直接使用BPE分词器解码，这样更准确
            try:
                return self.tokenizer.decode(indices)
            except Exception as e:
                print(f"⚠️ BPE解码失败: {e}")
                # 降级到逐个解码
                result = []
                for idx in indices:
                    if idx in self.itos:
                        result.append(self.itos[idx])
                    else:
                        result.append(f'<UNK_{idx}>')
                return ''.join(result)
        else:
            # 字符级解码
            return ''.join([self.itos.get(idx, '') for idx in indices])
    
    def get_batches(self, data: torch.Tensor, split: str, batch_size: int, 
                   context_window: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建批次数据"""
        # 切分训练集，验证集，测试集，比例为，训练80%，验证10%，测试10%
        train = data[:int(0.8 * len(data))]
        val = data[int(0.8 * len(data)): int(0.9 * len(data))]
        test = data[int(0.9 * len(data)):]

        # 选择对应的数据集
        batch_data = train
        if split == 'val':
            batch_data = val
        elif split == 'test':
            batch_data = test

        # 如果使用GPU，先将数据移动到GPU上，避免重复传输
        if device is not None and device.type == 'cuda' and batch_data.device != device:
            batch_data = batch_data.to(device)

        # 在GPU上生成随机索引，避免CPU到GPU的传输
        device_for_randint = device if device is not None else torch.device('cpu')
        ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,), device=device_for_randint)

        # 使用更高效的向量化索引方式，避免Python循环
        batch_indices = ix.unsqueeze(1) + torch.arange(context_window, device=ix.device).unsqueeze(0)
        target_indices = batch_indices + 1
        
        x = batch_data[batch_indices].long()
        y = batch_data[target_indices].long()

        # 确保数据在正确的设备上
        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        return x, y
    
    def get_stats(self) -> Dict:
        """获取数据处理统计信息"""
        stats = {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'dataset_size': len(self.dataset) if self.dataset is not None else 0,
            'data_type': str(self.dataset.dtype) if self.dataset is not None else None
        }
        
        if self.tokenizer_type == "chinese_bpe" and self.tokenizer:
            stats.update(self.tokenizer.get_compression_ratio())
        
        return stats
    
    def print_stats(self):
        """打印数据处理统计"""
        stats = self.get_stats()
        
        print("\n📊 数据处理统计:")
        print("=" * 40)
        print(f"分词器类型: {stats['tokenizer_type']}")
        print(f"词汇表大小: {stats['vocab_size']:,}")
        print(f"数据集大小: {stats['dataset_size']:,}")
        print(f"数据类型: {stats['data_type']}")
        
        if self.tokenizer_type == "chinese_bpe":
            print(f"压缩比: {stats.get('total_compression_ratio', 0):.2f} 字符/token")
            print(f"中文压缩比: {stats.get('chinese_compression_ratio', 0):.2f} 字符/token")


# 便利函数，保持向后兼容性
def download_and_prepare_data(data_file: str = "xiyouji.txt", force_download: bool = False, 
                            use_bpe: bool = True) -> Tuple[torch.Tensor, List, Dict, Dict]:
    """
    便利函数：下载并准备数据集
    
    Args:
        data_file: 数据文件名
        force_download: 是否强制重新下载
        use_bpe: 是否使用BPE分词器
        
    Returns:
        (dataset, vocab, itos, stoi)
    """
    tokenizer_type = "chinese_bpe" if use_bpe else "char_level"
    vocab_path = "chinese_tokenizer_vocab.json" if use_bpe else None
    
    processor = DataProcessor(tokenizer_type=tokenizer_type, vocab_path=vocab_path)
    result = processor.download_and_prepare_data(data_file, force_download)
    
    # 打印统计信息
    processor.print_stats()
    
    return result


def encode_text(text: str, stoi: Dict = None, processor: DataProcessor = None) -> List[int]:
    """编码文本为数字序列 - 兼容函数"""
    if processor:
        return processor.encode_text(text)
    elif stoi:
        return [stoi[ch] for ch in text if ch in stoi]
    else:
        raise ValueError("需要提供 processor 或 stoi 参数")


def decode_text(indices: List[int], itos: Dict = None, processor: DataProcessor = None) -> str:
    """解码数字序列为文本 - 兼容函数"""
    if processor:
        # 使用处理器的解码方法（推荐方式）
        return processor.decode_text(indices)
    elif itos:
        # 字符级解码的兼容方式
        return ''.join([itos.get(i, f'<UNK_{i}>') for i in indices])
    else:
        raise ValueError("需要提供 processor 或 itos 参数")


if __name__ == "__main__":
    print("🧪 测试数据处理模块...")
    
    # 测试BPE分词器
    print("\n1. 测试BPE分词器:")
    bpe_dataset, bpe_vocab, bpe_itos, bpe_stoi = download_and_prepare_data(use_bpe=True)
    
    # 测试字符级分词器
    print("\n2. 测试字符级分词器:")
    char_dataset, char_vocab, char_itos, char_stoi = download_and_prepare_data(use_bpe=False)
    
    # 比较结果
    print("\n📊 分词器比较:")
    print("=" * 40)
    print(f"BPE - 词汇表大小: {len(bpe_vocab):,}, 数据集大小: {len(bpe_dataset):,}")
    print(f"字符级 - 词汇表大小: {len(char_vocab):,}, 数据集大小: {len(char_dataset):,}")
    print(f"压缩效果: BPE比字符级减少了 {(1 - len(bpe_dataset)/len(char_dataset))*100:.1f}% 的token数量")
