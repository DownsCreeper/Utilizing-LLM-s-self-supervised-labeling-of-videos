"""
测试脚本：验证数据加载和训练框架
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_vid2seq import DatasetLoader, Vid2SeqTrainer


def test_data_loader():
    """测试数据加载器"""
    print("=" * 60)
    print("测试数据加载器（仅精细打标数据）")
    print("=" * 60)
    
    # 测试路径
    data_path = "dataset/breakfast/breakfast_5"
    
    if not Path(data_path).exists():
        print(f"⚠️  数据路径不存在: {data_path}")
        print("请确保数据集路径正确")
        return False
    
    try:
        # 创建数据加载器
        loader = DatasetLoader(data_path, split='train')
        
        print(f"✓ 成功加载数据集")
        print(f"  - 数据路径: {data_path}")
        print(f"  - 样本数量: {len(loader)}")
        
        # 显示进度文件信息
        if loader.fine_grained_videos:
            print(f"  - 精细打标视频: {loader.fine_grained_videos}")
        
        if len(loader) == 0:
            print("⚠️  数据集为空")
            print("    检查 breakfast_progress.json 是否存在")
            print("    检查 fine_grained.completed 列表是否为空")
            print("    检查视频文件夹是否包含精细标注文件（spark_labels_*.json）")
            return False
        
        # 测试获取单个样本
        sample = loader[0]
        print(f"\n样本示例:")
        print(f"  - 视频ID: {sample['video_id']}")
        print(f"  - 标注类型: {sample['label_type']}")
        print(f"  - 帧数: {sample['num_frames']}")
        print(f"  - 标注数量: {len(sample['labels'])}")
        print(f"  - 标注文件: {Path(sample['label_file']).name}")
        
        if sample['labels']:
            print(f"  - 第一个标注: {sample['labels'][0]}")
        
        # 测试批次生成
        print(f"\n测试批次生成...")
        batch_size = min(2, len(loader))
        batch_count = 0
        for batch in loader.get_batch(batch_size, shuffle=False):
            batch_count += 1
            print(f"  - 批次 {batch_count}: {len(batch)} 个样本")
            if batch_count >= 3:  # 只测试前3个批次
                break
        
        print(f"\n✓ 数据加载器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_setup():
    """测试训练器初始化"""
    print("\n" + "=" * 60)
    print("测试训练器初始化")
    print("=" * 60)
    
    try:
        # 创建测试配置
        config = {
            'num_epochs': 2,
            'batch_size': 2,
            'log_interval': 5,
            'val_interval': 1,
            'save_interval': 1,
            'output_dir': 'outputs/test_training',
            'model': {
                'num_layers': 6,
                'hidden_size': 256,
                'num_heads': 4,
                'vocab_size': 32128,
                'num_bins': 100,
            },
            'optimizer': {
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
            }
        }
        
        # 创建训练器
        trainer = Vid2SeqTrainer(config)
        print(f"✓ 训练器创建成功")
        
        # 测试设置数据
        data_path = "dataset/breakfast/breakfast_5"
        if Path(data_path).exists():
            trainer.setup_data(data_path)
            print(f"✓ 数据设置成功")
            print(f"  - 训练样本: {len(trainer.train_loader)}")
        else:
            print(f"⚠️  跳过数据设置（数据路径不存在）")
        
        # 测试设置模型
        trainer.setup_model()
        print(f"✓ 模型设置成功")
        print(f"  - 模型参数: {trainer.model_params}")
        
        # 检查输出目录
        print(f"✓ 输出目录创建成功")
        print(f"  - 检查点目录: {trainer.checkpoint_dir}")
        print(f"  - 日志目录: {trainer.log_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """测试迷你训练（2个epoch，用于验证流程）"""
    print("\n" + "=" * 60)
    print("测试迷你训练流程")
    print("=" * 60)
    
    data_path = "dataset/breakfast/breakfast_5"
    
    if not Path(data_path).exists():
        print(f"⚠️  跳过训练测试（数据路径不存在）")
        return False
    
    try:
        # 创建最小配置
        config = {
            'num_epochs': 2,
            'batch_size': 2,
            'log_interval': 5,
            'val_interval': 1,
            'save_interval': 1,
            'output_dir': 'outputs/test_mini_training',
            'model': {
                'num_layers': 4,
                'hidden_size': 128,
                'num_heads': 4,
                'vocab_size': 32128,
                'num_bins': 100,
            },
            'optimizer': {
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
            }
        }
        
        print("配置:")
        print(f"  - Epochs: {config['num_epochs']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - 输出目录: {config['output_dir']}")
        
        # 创建训练器
        trainer = Vid2SeqTrainer(config)
        trainer.setup_data(data_path)
        trainer.setup_model()
        
        print(f"\n开始迷你训练...")
        trainer.train()
        
        # 验证输出文件
        print(f"\n验证输出文件...")
        checkpoint_exists = (trainer.checkpoint_dir / 'checkpoint_epoch_1.pt').exists()
        log_exists = (trainer.log_dir / 'training_history.json').exists()
        
        print(f"  - 检查点文件: {'✓' if checkpoint_exists else '✗'}")
        print(f"  - 日志文件: {'✓' if log_exists else '✗'}")
        
        if checkpoint_exists and log_exists:
            print(f"\n✓ 迷你训练测试通过")
            return True
        else:
            print(f"\n⚠️  部分文件未生成")
            return False
        
    except Exception as e:
        print(f"✗ 迷你训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Vid2Seq 训练框架测试套件")
    print("=" * 60 + "\n")
    
    results = {
        "数据加载器": False,
        "训练器初始化": False,
        "迷你训练": False
    }
    
    # 运行测试
    results["数据加载器"] = test_data_loader()
    results["训练器初始化"] = test_trainer_setup()
    
    # 询问是否运行迷你训练测试
    print("\n" + "-" * 60)
    response = input("是否运行迷你训练测试（2个epoch）？这将创建实际的检查点文件。(y/n): ")
    if response.lower() in ['y', 'yes']:
        results["迷你训练"] = test_mini_training()
    else:
        print("跳过迷你训练测试")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败" if passed is False else "⊘ 跳过"
        print(f"{test_name}: {status}")
    
    all_passed = all(v is not False for v in results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("⚠️  部分测试未通过，请检查错误信息")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
