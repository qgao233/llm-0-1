#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL文件分割脚本
将大型JSONL文件按指定行数分割成多个小文件

使用方法:
    python split_jsonl.py input.jsonl                    # 默认每10万行分割一次
    python split_jsonl.py input.jsonl -n 50000          # 每5万行分割一次
    python split_jsonl.py input.jsonl -o output_dir     # 指定输出目录
    python split_jsonl.py input.jsonl --validate        # 分割时验证JSON格式
"""

import os
import json
import argparse
import sys
from pathlib import Path


def validate_json_line(line, line_num):
    """
    验证一行是否为有效的JSON格式
    
    Args:
        line (str): 要验证的行
        line_num (int): 行号
        
    Returns:
        bool: 是否为有效JSON
    """
    try:
        json.loads(line.strip())
        return True
    except json.JSONDecodeError as e:
        print(f"   ⚠️  第{line_num}行JSON格式错误: {str(e)[:50]}")
        return False


def get_output_filename(base_name, output_dir, part_num, total_parts=None):
    """
    生成输出文件名
    
    Args:
        base_name (str): 原文件名（不含扩展名）
        output_dir (str): 输出目录
        part_num (int): 分片编号
        total_parts (int): 总分片数（可选）
        
    Returns:
        str: 完整的输出文件路径
    """
    if total_parts and total_parts > 0:
        # 如果知道总分片数，使用固定宽度的编号
        width = len(str(total_parts))
        filename = f"{base_name}_part_{part_num:0{width}d}.jsonl"
    else:
        # 否则使用简单编号
        filename = f"{base_name}_part_{part_num:04d}.jsonl"
    
    return os.path.join(output_dir, filename)


def estimate_total_lines(file_path, sample_size=1000):
    """
    估算文件总行数（用于大文件）
    
    Args:
        file_path (str): 文件路径
        sample_size (int): 采样行数
        
    Returns:
        int: 估算的总行数
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # 对于小文件，直接计算
        if file_size < 100 * 1024 * 1024:  # 小于100MB
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        
        # 对于大文件，采样估算
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_lines = []
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                sample_lines.append(len(line.encode('utf-8')))
            
            if sample_lines:
                avg_line_size = sum(sample_lines) / len(sample_lines)
                estimated_lines = int(file_size / avg_line_size)
                return estimated_lines
            
        return 0
        
    except Exception as e:
        print(f"⚠️  无法估算文件行数: {e}")
        return 0


def split_jsonl_file(input_file, output_dir, lines_per_file=100000, validate=False):
    """
    分割JSONL文件
    
    Args:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录
        lines_per_file (int): 每个文件的行数
        validate (bool): 是否验证JSON格式
        
    Returns:
        dict: 分割结果统计
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取文件基本信息
    base_name = input_path.stem
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    
    print(f"📄 处理文件: {input_path.name}")
    print(f"📊 文件大小: {file_size_mb:.2f} MB")
    print(f"📁 输出目录: {output_dir}")
    print(f"📦 每文件行数: {lines_per_file:,}")
    
    # 估算总行数和分片数
    if file_size_mb > 10:  # 大于10MB才估算
        print("🔍 正在估算文件大小...")
        estimated_lines = estimate_total_lines(input_file)
        if estimated_lines > 0:
            estimated_parts = (estimated_lines + lines_per_file - 1) // lines_per_file
            print(f"📈 估算总行数: {estimated_lines:,}")
            print(f"📦 预计分片数: {estimated_parts}")
        else:
            estimated_parts = None
    else:
        estimated_parts = None
    
    print("=" * 60)
    
    # 开始分割
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'output_files': [],
        'empty_lines_skipped': 0
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            current_part = 1
            current_file = None
            current_file_lines = 0
            
            for line_num, line in enumerate(infile, 1):
                # 跳过空行
                if not line.strip():
                    stats['empty_lines_skipped'] += 1
                    continue
                
                stats['total_lines'] += 1
                
                # 验证JSON格式（如果启用）
                if validate:
                    if validate_json_line(line, line_num):
                        stats['valid_lines'] += 1
                    else:
                        stats['invalid_lines'] += 1
                        continue  # 跳过无效行
                else:
                    stats['valid_lines'] += 1
                
                # 检查是否需要创建新文件
                if current_file is None or current_file_lines >= lines_per_file:
                    # 关闭当前文件
                    if current_file is not None:
                        current_file.close()
                        print(f"✅ 完成分片 {current_part-1}: {current_file_lines:,} 行")
                    
                    # 创建新文件
                    output_filename = get_output_filename(
                        base_name, output_dir, current_part, estimated_parts
                    )
                    current_file = open(output_filename, 'w', encoding='utf-8')
                    stats['output_files'].append(output_filename)
                    current_file_lines = 0
                    
                    print(f"📝 开始写入分片 {current_part}: {os.path.basename(output_filename)}")
                    current_part += 1
                
                # 写入行到当前文件
                current_file.write(line)
                current_file_lines += 1
                
                # 显示进度
                if line_num % 10000 == 0:
                    print(f"   📊 已处理 {line_num:,} 行 (当前分片: {current_file_lines:,} 行)")
            
            # 关闭最后一个文件
            if current_file is not None:
                current_file.close()
                print(f"✅ 完成分片 {current_part-1}: {current_file_lines:,} 行")
                
    except UnicodeDecodeError:
        print("❌ 文件编码错误，尝试使用GBK编码...")
        try:
            # 重新尝试用GBK编码处理
            with open(input_file, 'r', encoding='gbk') as infile:
                # ... 重复上面的处理逻辑
                pass
        except Exception as e:
            raise Exception(f"无法读取文件: {e}")
            
    except Exception as e:
        if current_file:
            current_file.close()
        raise Exception(f"分割文件时出错: {e}")
    
    return stats


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="分割JSONL文件为多个小文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s data.jsonl                     # 默认每10万行分割
  %(prog)s data.jsonl -n 50000            # 每5万行分割
  %(prog)s data.jsonl -o ./split_files    # 指定输出目录
  %(prog)s data.jsonl --validate          # 验证JSON格式
  %(prog)s data.jsonl -n 200000 --validate -o output  # 组合参数
        """
    )
    
    parser.add_argument(
        'input_file',
        help='要分割的JSONL文件路径'
    )
    
    parser.add_argument(
        '-n', '--lines',
        type=int,
        default=100000,
        help='每个输出文件的行数（默认: 100000）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出目录（默认: 输入文件所在目录/文件名_split）'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='分割时验证JSON格式，跳过无效行'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    print("🚀 JSONL文件分割器启动")
    print("=" * 60)
    
    try:
        args = parse_arguments()
        
        # 确定输出目录
        if args.output:
            output_dir = args.output
        else:
            input_path = Path(args.input_file)
            output_dir = input_path.parent / f"{input_path.stem}_split"
        
        # 验证参数
        if args.lines <= 0:
            raise ValueError("每文件行数必须大于0")
        
        # 执行分割
        stats = split_jsonl_file(
            input_file=args.input_file,
            output_dir=output_dir,
            lines_per_file=args.lines,
            validate=args.validate
        )
        
        # 输出统计结果
        print("=" * 60)
        print("📊 分割完成统计:")
        print(f"   📝 总处理行数: {stats['total_lines']:,}")
        print(f"   ✅ 有效行数: {stats['valid_lines']:,}")
        
        if args.validate and stats['invalid_lines'] > 0:
            print(f"   ❌ 无效行数: {stats['invalid_lines']:,}")
        
        if stats['empty_lines_skipped'] > 0:
            print(f"   ⏭️  跳过空行: {stats['empty_lines_skipped']:,}")
        
        print(f"   📦 生成文件数: {len(stats['output_files'])}")
        print(f"   📁 输出目录: {output_dir}")
        
        # 显示生成的文件列表
        if len(stats['output_files']) <= 10:
            print("\n📄 生成的文件:")
            for file_path in stats['output_files']:
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   {os.path.basename(file_path)} ({file_size:.2f} MB)")
        else:
            print(f"\n📄 生成了 {len(stats['output_files'])} 个文件")
            print(f"   第一个: {os.path.basename(stats['output_files'][0])}")
            print(f"   最后一个: {os.path.basename(stats['output_files'][-1])}")
        
        print("\n✨ 分割完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
