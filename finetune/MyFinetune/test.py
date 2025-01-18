import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from langchain_huggingface import HuggingFaceEmbeddings
def test_model():
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型和分词器
    model_name = "train_output/20241122195936/epoch-1"
    # model_name = "unsloth/Qwen2.5-7B"
    print(f"加载模型: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 测试对话
    prompts = [
        "这家餐厅的服务态度很差，菜品也不新鲜。",
        "非常棒的购物体验，店员很热情，商品质量也很好！"
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
if __name__ == "__main__":
    test_model()

# outputs = model.generate(
#     **inputs,
#     # 1. 基本控制参数
#     max_new_tokens=512,      # 最多生成的token数量
#     min_new_tokens=10,       # 最少生成的token数量
    
#     # 2. 采样策略参数
#     temperature=0.7,         # 温度系数：越高越随机，越低越确定
#     top_k=50,               # 只保留概率最高的前k个token
#     top_p=0.9,              # 累积概率达到p的token才会被选择
    
#     # 3. 重复控制
#     repetition_penalty=1.1,  # 重复惩罚：>1降低重复，<1增加重复
#     no_repeat_ngram_size=3,  # 避免重复的n元组大小
    
#     # 4. 长度控制
#     max_length=2048,         # 总长度限制（包括输入）
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=tokenizer.eos_token_id,
    
#     # 5. 生成策略
#     do_sample=True,          # True使用采样，False使用贪婪搜索
#     num_beams=4,            # 束搜索的束宽
#     num_return_sequences=1,  # 返回多少个生成结果
    
#     # 6. 提前停止
#     early_stopping=True,     # 是否提前停止生成
        # )