#!/usr/bin/env python
"""
快速启动训练脚本
使用配置文件启动训练
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_vid2seq import Vid2SeqTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """合并配置"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description='Train Vid2Seq model using config file')
    parser.add_argument('--config', type=str, default='main/config_default.json',
                       help='Path to config file')
    parser.add_argument('--train_data', type=str, default=None,
                       help='Override training data path')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Override validation data path')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # 命令行参数覆盖
    overrides = {}
    if args.train_data:
        overrides.setdefault('dataset', {})['train_data'] = args.train_data
    if args.val_data:
        overrides.setdefault('dataset', {})['val_data'] = args.val_data
    if args.num_epochs:
        overrides.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.batch_size:
        overrides.setdefault('dataset', {})['batch_size'] = args.batch_size
    if args.output_dir:
        overrides.setdefault('output', {})['output_dir'] = args.output_dir
    
    # 合并配置
    config = merge_configs(config, overrides)
    
    # 构建训练器配置
    trainer_config = {
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['dataset']['batch_size'],
        'log_interval': config['training']['log_interval'],
        'val_interval': config['training']['val_interval'],
        'save_interval': config['training']['save_interval'],
        'output_dir': config['output']['output_dir'],
        'model': config['model'],
        'optimizer': {
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
        }
    }
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(json.dumps(trainer_config, indent=2))
    print("=" * 60 + "\n")
    
    # 创建训练器
    trainer = Vid2SeqTrainer(trainer_config)
    
    # 设置数据
    trainer.setup_data(
        config['dataset']['train_data'],
        config['dataset'].get('val_data')
    )
    
    # 设置模型
    trainer.setup_model()
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
