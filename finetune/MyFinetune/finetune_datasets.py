import torch
import datetime
import os
import logging
import json
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from torch.optim import AdamW
from accelerate import Accelerator
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

# 忽略警告
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")
warnings.filterwarnings("ignore", ".*MatMul8bitLt.*")

# 初始化常量
OUTPUT_DIR = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yelp_review_full"
BATCH_SIZE = 1  # 减小批量大小
EPOCHS = 5
LEARNING_RATE = 1e-6  # 降低学习率
GRADIENT_ACCUMULATION_STEPS = 64  # 增加梯度累积步数
MAX_LENGTH = 256  # 减小序列长度
SAVE_STEPS = 100
MAX_SAMPLES = 1000  # 用于测试
LOG_STEPS = 10
WARMUP_RATIO = 0.1
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.05

# 设备检查
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化加速器
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision="bf16",
    log_with="tensorboard",
    project_dir=OUTPUT_DIR
)

def init_logger(log_dir: str = "./") -> logging.Logger:
    """初始化日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = [
        logging.FileHandler(
            f'{log_dir}training.{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
            mode='w',
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ReviewDataset(Dataset):
    """评论数据集类"""
    def __init__(self, dataset, tokenizer, max_length=MAX_LENGTH):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.process_item(item)

    def process_item(self, item):
        """处理单个数据项"""
        # 如果已经是处理好的格式，直接返回
        if "conversations" in item:
            return item
        
        # 只有原始数据才需要处理
        text = item["text"]
        label = item["label"]
        
        # 构建对话格式
        conversation = [
            {
                "role": "system",
                "content": "你是一个专业的评论分析助手。你需要分析评论的情感倾向，并给出详细的分析理由。"
            },
            {
                "role": "user",
                "content": f"请分析这条评论的情感倾向：{text}"
            },
            {
                "role": "assistant",
                "content": self.generate_response(text, label)
            }
        ]
        
        return {"conversations": conversation}

    def generate_response(self, text: str, label: int) -> str:
        """生成助手回复"""
        stars = "⭐" * (label + 1)
        sentiment = "积极" if label >= 3 else "消极"
        
        response = (
            f"这是一个{stars}（{label + 1}星）评论，整体表达了{sentiment}的情感。\n\n"
            f"详细分析：\n"
            f"1. 评分：{label + 1}星级评价\n"
            f"2. 情感倾向：{sentiment}\n"
            f"3. 关键内容：{text}\n"
            f"4. 分析结论：根据评论内容和评分，这条评论明确表达了{sentiment}的态度。"
        )
        return response

    @staticmethod
    def collate_fn(batch, tokenizer):
        """数据批处理函数"""
        conversations = [item["conversations"] for item in batch]
        
        # 使用tokenizer的chat模板
        texts = [tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True
        ) for conv in conversations]
        
        # 编码输入
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        labels = inputs["input_ids"].clone()
        
        # 设置非助手回复部分的标签为-100
        for i, text in enumerate(texts):
            assistant_start = text.find("<|im_start|>assistant")
            if assistant_start != -1:
                assistant_tokens = tokenizer(text[assistant_start:], add_special_tokens=False)
                labels[i, :len(inputs["input_ids"][i]) - len(assistant_tokens["input_ids"])] = -100
        
        return inputs, labels

def prepare_dataset(tokenizer):
    """准备数据集"""
    cache_path = "cache/processed_dataset"
    
    if os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
    else:
        # 加载原始数据集
        dataset = load_dataset(DATASET_NAME)
        
        if MAX_SAMPLES:
            dataset["train"] = dataset["train"].select(range(MAX_SAMPLES))
        
        # 保存处理后的数据集
        dataset.save_to_disk(cache_path)
    
    # 创建训练集和验证集
    train_size = int(len(dataset["train"]) * (1 - VALIDATION_SPLIT))
    val_size = len(dataset["train"]) - train_size
    
    train_dataset = ReviewDataset(
        dataset["train"].select(range(train_size)), 
        tokenizer
    )
    val_dataset = ReviewDataset(
        dataset["train"].select(range(train_size, len(dataset["train"]))),
        tokenizer
    )
    
    return train_dataset, val_dataset

def evaluate(model, val_loader, tokenizer, logger):
    """评估函数"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train():
    """训练主函数"""
    logger = init_logger(OUTPUT_DIR)
    logger.info(f"设备信息: {device}")
    
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # 改用 float16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型和分词器
    logger.info(f"加载模型：{MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.to(device)
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.to(device)
    

    # 准备数据集
    logger.info("准备数据集")
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: ReviewDataset.collate_fn(batch, tokenizer),
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: ReviewDataset.collate_fn(batch, tokenizer),
        num_workers=0,
        pin_memory=True
    )
    
    # 优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8  # 添加 eps 参数
    )
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 准备训练
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    logger.info("开始训练")
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                total_loss += loss.item()
                
                if global_step % LOG_STEPS == 0:
                    avg_loss = total_loss / LOG_STEPS
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    total_loss = 0
            
            # 定期清理缓存
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 验证
        val_loss = evaluate(model, val_loader, tokenizer, logger)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            logger.info("保存最佳模型")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{OUTPUT_DIR}/best_model",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered")
                break
        
        # 每个epoch结束后保存
        logger.info(f"保存 Epoch {epoch+1} 模型")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{OUTPUT_DIR}/epoch-{epoch+1}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch-{epoch+1}")
    
    logger.info("训练完成，保存最终模型")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{OUTPUT_DIR}/final_model",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

def test_model(model_path: str, test_prompt: str) -> str:
    """测试模型"""
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 构建系统提示
    prompt = f"""<|im_start|>system
你是一个专业的评论分析助手。你需要分析评论的情感倾向，并给出详细的分析理由。
<|im_end|>
<|im_start|>user
请分析这条评论的情感倾向：{test_prompt}
<|im_end|>
<|im_start|>assistant"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # 减小生成长度
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

if __name__ == "__main__":
    # 训练模型
    train()
    
    # 测试模型
    test_prompts = [
        "这家餐厅的服务态度很差，菜品也不新鲜。",
        "非常棒的购物体验，店员很热情，商品质量也很好！"
    ]
    
    # 优先使用最佳模型
    if os.path.exists(f"{OUTPUT_DIR}/best_model"):
        model_path = f"{OUTPUT_DIR}/best_model"
    else:
        model_path = f"{OUTPUT_DIR}/final_model"
    
    print("\n测试结果:")
    for prompt in test_prompts:
        response = test_model(model_path, prompt)
        print(f"\n提示: {prompt}")
        print(f"回复: {response}")