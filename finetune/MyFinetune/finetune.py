import torch
import json
import datetime
import os
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from torch.optim import AdamW
from accelerate import Accelerator
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm.auto import tqdm

# 忽略特定警告
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")
warnings.filterwarnings("ignore", ".*MatMul8bitLt.*")

# 检查GPU是否可用
if torch.cuda.is_available():
    device = "cuda"
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"当前GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = "cpu"
    print("未检测到GPU，使用CPU训练")

# 初始化常量
OUTPUT_DIR = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024
SAVE_STEPS = 50
WARMUP_STEPS = 100
TRAIN_DATA_PATH = ["data/data1.json"]
LOG_STEPS = 10

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

class InstructionDataset(Dataset):
    """指令数据集类"""
    def __init__(self, data_path: List[str]):
        super().__init__()
        self.data = []
        # 加载多个文件
        for path in data_path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if not isinstance(item.get("conversations"), list):
                        continue
                    # 添加系统提示
                    if not any(conv["role"] == "system" for conv in item["conversations"]):
                        item["conversations"].insert(0, {
                            "role": "system",
                            "content": "你是一个评论分析助手。"
                        })
                    self.data.append(item)
        self._validate_data()
    
    def _validate_data(self):
        """验证数据格式"""
        for item in self.data:
            assert "conversations" in item, "每个数据项必须包含conversations字段"
            assert len(item["conversations"]) >= 2, "每个对话必须至少包含一问一答"
            for conv in item["conversations"]:
                assert "role" in conv, "每条消息必须包含role字段"
                assert "content" in conv, "每条消息必须包含content字段"
                assert conv["role"] in ["user", "assistant", "system"], "role必须是user、assistant或system"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def find_assistant_content_indexes(tokenized_ids: List[int], tokenizer: AutoTokenizer) -> List[Tuple[int, int]]:
    """在tokenized_ids中找出助手回复内容的起始和结束位置"""
    assistant_start = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    start_indexes = []
    end_indexes = []
    
    for i in range(len(tokenized_ids) - len(assistant_start)):
        if tokenized_ids[i:i+len(assistant_start)] == assistant_start:
            start_indexes.append(i)
            for j in range(i + len(assistant_start), len(tokenized_ids)):
                if tokenized_ids[j:j+len(end_token)] == end_token:
                    end_indexes.append(j + len(end_token))
                    break
    
    return list(zip(start_indexes, end_indexes))

def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """数据批次处理函数"""
    conversations = [item["conversations"] for item in batch]
    texts = [tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True
    ) for conv in conversations]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    labels = inputs["input_ids"].clone()
    
    for i, ids in enumerate(inputs["input_ids"].tolist()):
        label_ids = [-100] * len(ids)
        for start, end in find_assistant_content_indexes(ids, tokenizer):
            label_ids[start:end] = ids[start:end]
        labels[i] = torch.tensor(label_ids, dtype=torch.long)
    
    return inputs, labels

def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    step: Optional[int] = None,
    is_final: bool = False
):
    """保存模型检查点
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
        step: 当前步数（可选）
        is_final: 是否为最终模型
    """
    if is_final:
        save_dir = os.path.join(output_dir, "final_model")
    else:
        save_dir = output_dir if step is None else os.path.join(output_dir, f"checkpoint-{step}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)

def train():
    """训练主函数"""
    logger = init_logger(OUTPUT_DIR)
    logger.info(f"设备信息: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    logger.info(f"加载模型：{MODEL_NAME}")
    
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        quantization_config=quantization_config
    )
    model.to(device)
    
    # 配置模型参数
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.to(device)
    
    # 确保tokenizer有必要的token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    logger.info("加载训练数据集")
    train_dataset = InstructionDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, device),
        num_workers=0  # 暂时设为0，避免多进程问题
    )
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    # 准备训练
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    logger.info(f"模型所在设备: {next(model.parameters()).device}")
    logger.info("开始训练循环")
    
    global_step = 0
    model.train()
    
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                total_loss += loss.item()
                
                if global_step % LOG_STEPS == 0:
                    avg_loss = total_loss / LOG_STEPS
                    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Step {global_step}, Loss: {avg_loss:.4f}")
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    total_loss = 0
                
                if global_step % SAVE_STEPS == 0:
                    logger.info(f"保存检查点 step {global_step}")
                    save_checkpoint(model, tokenizer, OUTPUT_DIR, global_step)
                
                progress_bar.update(1)
        
        logger.info(f"Epoch {epoch+1} 完成")
        
        # 每个epoch结束后保存
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{OUTPUT_DIR}/epoch-{epoch+1}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch-{epoch+1}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("训练完成，保存最终模型")
    save_checkpoint(model, tokenizer, OUTPUT_DIR, is_final=True)

def test_model(model_path: str, test_prompt: str) -> str:
    """测试微调后的模型"""
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
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
    full_prompt = f"""<|im_start|>system
你是一个评论分析助手。
<|im_end|>
<|im_start|>user
{test_prompt}
<|im_end|>
<|im_start|>assistant"""
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
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
    
    model_path = os.path.join(OUTPUT_DIR, "final_model")
    print("\n测试结果:")
    for prompt in test_prompts:
        response = test_model(model_path, prompt)
        print(f"\n提示: {prompt}")
        print(f"回复: {response}")