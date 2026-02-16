#!/usr/bin/env python3
"""
合并v4.1训练数据并应用权重
权重方案：
- 禅宗祖师: 3x
- 佛学名词: 2x  
- 经典问答: 2x
- 幻觉控制: 1x (降权，从v3继承)
- 边界拒答: 2x (从v3继承)
- 身份强化: 2x (从v3继承)
"""
import json
import os

BASE_DIR = os.path.expanduser('~/code/buddhist-72b-distill')
V4_DATA_DIR = os.path.join(BASE_DIR, 'v4_training/data/v4.1')
V3_TRAIN_FILE = os.path.join(BASE_DIR, 'data/train/buddhist_v3_train.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/train/buddhist_v4.1_train.json')

# 权重配置
WEIGHTS = {
    'patriarch_qa.jsonl': 3,  # 禅宗祖师 3x
    'term_qa.jsonl': 2,       # 佛学名词 2x
    'sutra_qa.jsonl': 2,      # 经典问答 2x
}

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    all_data = []
    
    # 1. 加载v3数据（包含幻觉控制、边界拒答、身份强化等）
    print(f"Loading v3 data from {V3_TRAIN_FILE}...")
    v3_data = load_json(V3_TRAIN_FILE)
    print(f"  Loaded {len(v3_data)} samples from v3")
    
    # 分类v3数据
    hallucination_data = []  # 幻觉控制
    boundary_data = []       # 边界拒答
    identity_data = []       # 身份强化
    other_data = []          # 其他
    
    hallucination_keywords = ['不在我的专业范围', '超出了我的专业范围', '无法查证', '我无法确认']
    boundary_keywords = ['股票', '彩票', '投资', '政治', '选举']
    identity_keywords = ['善知识', '我是', '我的身份']
    
    for item in v3_data:
        inst = item.get('instruction', '')
        out = item.get('output', '')
        
        if any(kw in out for kw in hallucination_keywords):
            hallucination_data.append(item)
        elif any(kw in inst for kw in boundary_keywords):
            boundary_data.append(item)
        elif any(kw in inst for kw in identity_keywords) or any(kw in out for kw in identity_keywords):
            identity_data.append(item)
        else:
            other_data.append(item)
    
    print(f"  v3 breakdown: hallucination={len(hallucination_data)}, boundary={len(boundary_data)}, identity={len(identity_data)}, other={len(other_data)}")
    
    # 应用v3权重（幻觉控制降权1x，边界拒答2x，身份强化2x，其他1x）
    all_data.extend(hallucination_data)  # 1x
    all_data.extend(boundary_data * 2)   # 2x
    all_data.extend(identity_data * 2)   # 2x
    all_data.extend(other_data)          # 1x
    
    print(f"  After v3 weighting: {len(all_data)} samples")
    
    # 2. 加载v4.1新数据并应用权重
    print(f"\nLoading v4.1 data from {V4_DATA_DIR}...")
    for filename, weight in WEIGHTS.items():
        filepath = os.path.join(V4_DATA_DIR, filename)
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            # 转换格式：添加input字段
            formatted_data = []
            for item in data:
                formatted_data.append({
                    'instruction': item['instruction'],
                    'input': '',
                    'output': item['output']
                })
            # 应用权重
            weighted_data = formatted_data * weight
            all_data.extend(weighted_data)
            print(f"  {filename}: {len(data)} x {weight} = {len(weighted_data)} samples")
        else:
            print(f"  {filename}: NOT FOUND")
    
    # 3. 保存合并后的数据
    print(f"\nTotal samples: {len(all_data)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {OUTPUT_FILE}")
    
    # 4. 统计
    print(f"\n=== V4.1 Training Data Summary ===")
    print(f"Total samples: {len(all_data)}")
    print(f"Output file size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
