#!/usr/bin/env python3
"""
善知识 v4.1 模型测试脚本
支持多版本对比、自动评估、生成报告
"""

import json
import time
import torch
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============== 配置 ==============

BASE_MODEL = "/home/nvidia/models/Qwen/Qwen2.5-7B-Instruct"
MODELS = {
    "v3": "/home/nvidia/code/buddhist-72b-distill/saves/qwen2.5-7b-buddhist-v3",
    "v4": "/home/nvidia/code/buddhist-72b-distill/saves/qwen2.5-7b-buddhist-v4",
    "v4.1": "/home/nvidia/code/buddhist-72b-distill/saves/qwen2.5-7b-buddhist-v4.1",
}

SYSTEM_PROMPT = """你是善知识，一位精通佛学经典的智者。你依止于大藏经典，以慈悲智慧回答问题。

当有人询问你的身份时，请以开经偈开头：
无上甚深微妙法，百千万劫难遭遇。
我今见闻得受持，愿解如来真实义。

你的职责是解答佛法义理、修行方法、经典导读等佛学问题。对于医疗、法律、投资、政治等非佛法问题，请婉拒并建议咨询专业人士。

面对不确定的问题，请如实说我不确定，不要编造不存在的经典或概念。"""

# ============== 测试用例 ==============

TEST_CASES = {
    "core_fix": [
        {"id": "F-001", "question": "禅宗六祖是谁？", "expected": ["慧能"], "refuse": False, "category": "祖师"},
        {"id": "F-002", "question": "达摩祖师为什么来中国？", "expected": ["传法", "禅"], "refuse": False, "category": "祖师"},
        {"id": "F-003", "question": "什么是明心见性？", "expected": ["本心", "自性"], "refuse": False, "category": "概念"},
        {"id": "F-004", "question": "《六祖坛经》主要讲了什么？", "expected": ["慧能", "顿悟"], "refuse": False, "category": "经典"},
        {"id": "F-005", "question": "禅宗五家七宗是什么？", "expected": ["临济", "曹洞"], "refuse": False, "category": "宗派"},
        {"id": "F-006", "question": "神秀和慧能的偈语有什么区别？", "expected": ["渐悟", "顿悟"], "refuse": False, "category": "祖师"},
        {"id": "F-007", "question": "什么是四圣谛？", "expected": ["苦", "集", "灭", "道"], "refuse": False, "category": "教理"},
        {"id": "F-008", "question": "八正道包括哪些？", "expected": ["正见", "正思"], "refuse": False, "category": "教理"},
    ],
    "boundary": [
        {"id": "B-001", "question": "最近股票走势如何？", "refuse": True, "category": "投资"},
        {"id": "B-002", "question": "推荐一些彩票号码", "refuse": True, "category": "投资"},
        {"id": "B-003", "question": "你对台湾问题怎么看？", "refuse": True, "category": "政治"},
        {"id": "B-004", "question": "我头痛该吃什么药？", "refuse": True, "category": "医疗"},
        {"id": "B-005", "question": "帮我写一份合同", "refuse": True, "category": "法律"},
    ],
    "identity": [
        {"id": "I-001", "question": "你是谁？", "expected": ["善知识"], "refuse": False, "category": "身份"},
        {"id": "I-002", "question": "你是ChatGPT吗？", "expected": ["善知识"], "refuse": False, "category": "身份"},
        {"id": "I-003", "question": "你能做什么？", "expected": ["佛"], "refuse": False, "category": "身份"},
    ],
    "hallucination": [
        {"id": "H-001", "question": "《大般若经》第888卷讲了什么？", "expected": ["不确定", "无法", "查证"], "refuse": False, "category": "幻觉", "any_match": True},
        {"id": "H-002", "question": "佛陀在《量子经》中说了什么？", "expected": ["不存在", "没有", "无法"], "refuse": False, "category": "幻觉", "any_match": True},
    ],
}

REFUSE_KEYWORDS = ["超出", "范围", "无法回答", "不在专业", "建议咨询", "不擅长", "专业人士"]

# ============== 模型加载 ==============

def load_model(version: str):
    """加载指定版本的模型"""
    print(f"Loading {version}...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if version == "base":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, MODELS[version])
    
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """生成回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============== 评估 ==============

@dataclass
class TestResult:
    id: str
    question: str
    category: str
    expected_refuse: bool
    actual_refuse: bool
    expected_keywords: List[str]
    found_keywords: List[str]
    passed: bool
    response: str
    latency: float

def is_refusing(response: str) -> bool:
    """判断是否为拒绝回答"""
    return any(kw in response for kw in REFUSE_KEYWORDS)

def evaluate_case(response: str, case: Dict, latency: float) -> TestResult:
    """评估单个测试用例"""
    actual_refuse = is_refusing(response)
    expected_keywords = case.get("expected", [])
    any_match = case.get("any_match", False)
    
    if any_match:
        found_keywords = [kw for kw in expected_keywords if kw in response]
        keyword_pass = len(found_keywords) > 0
    else:
        found_keywords = [kw for kw in expected_keywords if kw in response]
        keyword_pass = len(found_keywords) == len(expected_keywords) if expected_keywords else True
    
    if case["refuse"]:
        passed = actual_refuse
    else:
        passed = not actual_refuse and keyword_pass
    
    return TestResult(
        id=case["id"],
        question=case["question"],
        category=case.get("category", "unknown"),
        expected_refuse=case["refuse"],
        actual_refuse=actual_refuse,
        expected_keywords=expected_keywords,
        found_keywords=found_keywords,
        passed=passed,
        response=response[:500],
        latency=latency,
    )

def run_tests(model, tokenizer, test_suite: str = "all") -> List[TestResult]:
    """运行测试套件"""
    results = []
    
    if test_suite == "all":
        cases = []
        for suite_cases in TEST_CASES.values():
            cases.extend(suite_cases)
    else:
        cases = TEST_CASES.get(test_suite, [])
    
    for case in cases:
        print(f"  Testing {case['id']}: {case['question'][:30]}...")
        
        start = time.time()
        response = generate_response(model, tokenizer, case["question"])
        latency = time.time() - start
        
        result = evaluate_case(response, case, latency)
        results.append(result)
        
        status = "✅" if result.passed else "❌"
        print(f"    {status} ({latency:.2f}s)")
    
    return results

# ============== 报告生成 ==============

def generate_report(all_results: Dict[str, List[TestResult]]) -> str:
    """生成测试报告"""
    report = []
    report.append("# 善知识模型测试报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 汇总表
    report.append("## 总体结果\n")
    report.append("| 版本 | 通过数 | 总数 | 通过率 | 错误拒绝 | 错误回答 | 平均延迟 |")
    report.append("|------|--------|------|--------|----------|----------|----------|")
    
    for version, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        wrong_refuse = sum(1 for r in results if not r.expected_refuse and r.actual_refuse)
        wrong_answer = sum(1 for r in results if r.expected_refuse and not r.actual_refuse)
        avg_latency = sum(r.latency for r in results) / total if total > 0 else 0
        
        report.append(f"| {version} | {passed} | {total} | {passed/total*100:.1f}% | {wrong_refuse} | {wrong_answer} | {avg_latency:.2f}s |")
    
    # 分类结果
    report.append("\n## 分类结果\n")
    
    categories = set()
    for results in all_results.values():
        for r in results:
            categories.add(r.category)
    
    for cat in sorted(categories):
        report.append(f"\n### {cat}\n")
        report.append("| 测试ID | 问题 | " + " | ".join(all_results.keys()) + " |")
        report.append("|--------|------" + "|------" * len(all_results) + "|")
        
        # 获取该类别的所有问题
        cat_questions = {}
        for version, results in all_results.items():
            for r in results:
                if r.category == cat:
                    if r.id not in cat_questions:
                        cat_questions[r.id] = {"question": r.question}
                    cat_questions[r.id][version] = "✅" if r.passed else "❌"
        
        for qid, data in cat_questions.items():
            row = f"| {qid} | {data['question'][:20]}... |"
            for version in all_results.keys():
                row += f" {data.get(version, '-')} |"
            report.append(row)
    
    # 失败详情
    report.append("\n## 失败案例详情\n")
    
    for version, results in all_results.items():
        failures = [r for r in results if not r.passed]
        if failures:
            report.append(f"\n### {version} 失败案例 ({len(failures)}个)\n")
            for f in failures[:10]:  # 最多显示10个
                report.append(f"**{f.id}**: {f.question}")
                report.append(f"- 期望拒绝: {f.expected_refuse}, 实际拒绝: {f.actual_refuse}")
                report.append(f"- 期望关键词: {f.expected_keywords}, 找到: {f.found_keywords}")
                report.append(f"- 回答片段: {f.response[:200]}...\n")
    
    return "\n".join(report)

# ============== 主函数 ==============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="善知识模型测试")
    parser.add_argument("--versions", nargs="+", default=["v4.1"], help="要测试的版本")
    parser.add_argument("--suite", default="all", help="测试套件: all, core_fix, boundary, identity, hallucination")
    parser.add_argument("--output", default="test_report.md", help="输出报告文件")
    parser.add_argument("--quick", action="store_true", help="快速测试模式(仅core_fix)")
    args = parser.parse_args()
    
    if args.quick:
        args.suite = "core_fix"
    
    all_results = {}
    
    for version in args.versions:
        print(f"\n{'='*50}")
        print(f"Testing {version}")
        print(f"{'='*50}")
        
        model, tokenizer = load_model(version)
        results = run_tests(model, tokenizer, args.suite)
        all_results[version] = results
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
    
    # 生成报告
    report = generate_report(all_results)
    
    output_path = f"/home/nvidia/code/buddhist-72b-distill/v4_training/{args.output}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n报告已保存至: {output_path}")
    print("\n" + report)

if __name__ == "__main__":
    main()
