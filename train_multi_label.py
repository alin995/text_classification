import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

data_file = "./data.json" # 数据文件路径，数据需要提前下载
model_name = "hfl/chinese-bert-wwm-ext" # 所使用模型

# 加载数据集
dataset = load_dataset("json", data_files=data_file, cache_dir='cache')
dataset = dataset["train"]
dataset = dataset.filter(lambda x: x["title"] is not None)

# 数据集处理
tokenizer = AutoTokenizer.from_pretrained(model_name)
def expand_labels(indices, size):
  r = np.zeros(size, dtype=float)
  for x in indices:
    r[x] = .999
  return list(r.astype(float))

def normalize_dataset(sample):
  normalized_sample = tokenizer(sample["title"], max_length=512, truncation=True)
  labels = np.array([expand_labels(x, 85) for x in sample["label"]]).astype(float)
  normalized_sample['label'] = labels
  return normalized_sample

dataset = dataset.map(normalize_dataset, batched=True)

def process_function(examples):
  tokenized_examples = tokenizer(examples["title"], max_length=64, truncation=True)
  tokenized_examples["labels"] = examples["label"]
  return tokenized_examples



# 构建评估函数
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  fn_loss = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
  loss = fn_loss(torch.from_numpy(predictions), torch.from_numpy(labels))
  return {"loss": loss}

# 训练器配置
model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=85)

args = TrainingArguments(
  learning_rate=2e-5,
  per_device_train_batch_size=32,
  per_device_eval_batch_size=128,
  num_train_epochs=20,
  weight_decay=0.01,
  output_dir="model_for_seqclassification",
  logging_steps=200,
  evaluation_strategy="steps",
  save_strategy="steps",
  save_steps=200,
  load_best_model_at_end=True,
  metric_for_best_model="loss",
  fp16=False
)

trainer = Trainer(
  model,
  args,
  train_dataset=dataset,
  eval_dataset=dataset,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics,
  data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# 训练与评估
trainer.train()

#trainer.evaluate()
