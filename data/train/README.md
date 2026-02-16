# 训练数据

完整的训练数据文件 (`buddhist_v4.1_train.json`, 28MB) 未包含在Git仓库中。

## 获取方式

### 方式1: 自行生成

```bash
# 1. 生成v4.1专项数据
python scripts/generate_data.py

# 2. 合并数据
python scripts/merge_data.py
```

### 方式2: 从Release下载

训练数据将包含在GitHub Release中。

## 数据格式

```json
[
  {
    "instruction": "禅宗六祖是谁？",
    "input": "",
    "output": "禅宗六祖是慧能大师..."
  }
]
```

## 数据统计

| 版本 | 条数 | 文件大小 |
|------|------|----------|
| v4.1 | 14,037 | ~28MB |
