from datasets import load_dataset
import json

def explore_dataset():
    """探索数据集结构"""
    print("正在加载数据集...")
    dataset = load_dataset("yelp_review_full")
    
    print("\n数据集基本信息:")
    print("="*50)
    print(f"数据集分片: {dataset.keys()}")
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"测试集大小: {len(dataset['test'])}")
    
    print("\n数据集字段信息:")
    print("="*50)
    print(f"可用字段: {dataset['train'].features}")
    
    print("\n样本数据:")
    print("="*50)
    # 打印前3个样本
    for i in range(3):
        sample = dataset['train'][i]
        print(f"\n样本 {i+1}:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        
    # 检查数据类型
    print("\n数据类型信息:")
    print("="*50)
    sample = dataset['train'][0]
    for key, value in sample.items():
        print(f"{key}: {type(value)}")

if __name__ == "__main__":
    explore_dataset() 