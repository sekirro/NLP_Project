import torch
from safetensors import safe_open

def compare_models(original_path: str, finetuned_path: str):
    """比较原始模型和微调后模型的参数"""
    # 加载两个模型的参数
    original = safe_open(f"{original_path}/model.safetensors", framework="pt", device="cpu")
    finetuned = safe_open(f"{finetuned_path}/model.safetensors", framework="pt", device="cpu")
    
    print("参数差异:")
    for key in original.keys():
        if key in finetuned.keys():
            # 计算参数差异
            orig_tensor = original.get_tensor(key)
            fine_tensor = finetuned.get_tensor(key)
            diff = torch.abs(orig_tensor - fine_tensor).mean().item()
            print(f"{key}: 平均差异 = {diff:.6f}")

# 使用示例
compare_models(
    "C:/Users/13686/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",  # 原始模型路径
    "train_output/20241121154403/final_model"  # 微调后的模型路径
)