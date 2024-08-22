import argparse
import json
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def train_core(split_datapaths, name_or_path, device, dtype,train_args):
    train_args=json.loads(train_args)
    dataset = load_from_disk(split_datapaths)
    print("Training dataset loaded!")
    if dtype == "float32":
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True, device_map=device
        )
    elif dtype == "float16":
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True, device_map=device
        ).half()
    elif dtype == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16
        )
    elif dtype == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True, device_map=device, quantization_config=quantization_config
        )
    elif dtype == "int4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True, device_map=device, quantization_config=quantization_config
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    print("CausalLM is loaded!")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True,device_map=device)
    print("tokenizer is loaded!")
    # 设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    def get_max_length(dataset, context_key='context', question_key='question'):
        max_length = 0
        for example in dataset:
            context = example[context_key] if context_key in example else ""
            question = example[question_key] if question_key in example else ""
            context_length = len(context.split()) if isinstance(context, str) else 0
            question_length = len(question.split()) if isinstance(question, str) else 0
            total_length = context_length + question_length
            if total_length > max_length:
                max_length = total_length
        return max_length
    max_length = get_max_length(dataset['train'])
    # 预处理函数
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            # If the answer is not fully inside the context, label it (0, 0)
            if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                start_token_idx = None
                end_token_idx = None
                for idx, (start, end) in enumerate(offset):
                    if start <= start_char < end:
                        start_token_idx = idx
                    if start < end_char <= end:
                        end_token_idx = idx
                        break
                if start_token_idx is not None and end_token_idx is not None:
                    start_positions.append(start_token_idx)
                    end_positions.append(end_token_idx)
                else:
                    start_positions.append(0)
                    end_positions.append(0)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # 预处理数据集
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    print("Tokenized datasets are generated!")
    # 设置训练参数
    training_args = TrainingArguments(
        **train_args
    )

    # 定义数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Start training! Please wait a moment.")
    # 开始训练
    trainer.train()
    print("The training is over and the model is being evaluated!")
    # 评估模型
    results = trainer.evaluate()

    # 返回评估结果字符串
    eval_summary = f"模型评估结果:\n准确率: {results['accuracy']:.2f}\n损失: {results['loss']:.2f}\n其他指标: {results['other_metrics']}"
    print(eval_summary)
    return eval_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train core model')
    parser.add_argument('--split_datapaths', type=str, required=True, help='Path to split data')
    parser.add_argument('--name_or_path', type=str, required=True, help='Model name or path')
    parser.add_argument('--device', type=str, required=True, help='Device to use for training')
    parser.add_argument('--dtype', type=str, required=True, help='Data type')
    parser.add_argument('--train_args', type=str, required=True, help='Additional training arguments')

    args = parser.parse_args()

    train_core(args.split_datapaths, args.name_or_path, args.device, args.dtype, args.train_args)
