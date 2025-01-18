import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", ".*MatMul8bitLt.*")

def test_model():
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型和分词器
    model_name = "D:/Desktop/finetune/train_output/20241122203104/final_model"
    print(f"加载模型: {model_name}")
    
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 测试对话
    prompts = [
        "这家餐厅的服务态度很差，菜品也不新鲜。",
        "非常棒的购物体验，店员很热情，商品质量也很好！"
    ]

    # 预热一次模型（可选）
    with torch.no_grad():
        warmup_prompt = "测试评论"
        warmup_input = tokenizer(warmup_prompt, return_tensors="pt").to(device)
        model.generate(**warmup_input, max_new_tokens=1)

    for prompt in prompts:
        print(f"\n用户: {prompt}")
        
        # 构建完整提示
        full_prompt = f"""<|im_start|>system
你是一个评论分析助手。
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant"""
        
        # 生成回复
        with torch.no_grad():  # 禁用梯度计算加快推理
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码并提取回复
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        
        print(f"助手: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()