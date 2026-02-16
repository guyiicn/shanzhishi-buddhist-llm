#!/usr/bin/env python3
"""
v4.1 Training Data Generator for 善知识 Buddhist AI
Generates: patriarch_qa.jsonl, term_qa.jsonl, sutra_qa.jsonl
Uses: Qdrant RAG + Qwen-Plus LLM
"""

import asyncio
import json
import os
import sys
import time
import httpx
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient

# ============================================================
# Configuration
# ============================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "cbeta"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3:latest"
DASHSCOPE_API_KEY = "sk-8cdb59cb607348ff9ef65478d24d635b"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

OUTPUT_DIR = Path(os.path.expanduser("~/code/buddhist-72b-distill/v4_training/data/v4.1"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RAG Services
# ============================================================
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama bge-m3"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

async def rag_search(query: str, top_k: int = 8, filters: Optional[Dict] = None) -> List[Dict]:
    """Search RAG for relevant content"""
    vec = await get_embedding(query)
    
    query_filter = None
    if filters:
        from qdrant_client import models
        must = []
        for k, v in filters.items():
            must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        query_filter = models.Filter(must=must)
    
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    )
    
    return [
        {
            "content": p.payload.get("content", ""),
            "score": p.score,
            "metadata": {k: v for k, v in p.payload.items() if k not in ["content", "doc_id"]}
        }
        for p in results.points
    ]

async def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call Qwen-Plus via DashScope API"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DASHSCOPE_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ============================================================
# Data Definitions
# ============================================================

PATRIARCH_TOPICS = [
    # 达摩大师 (初祖) - 8+ questions
    {"figure": "达摩大师", "questions": [
        "达摩祖师是谁？请介绍他的生平。",
        "达摩祖师为什么要从印度来到中国？",
        "达摩祖师面壁九年的故事是怎样的？",
        "达摩祖师与梁武帝的对话是什么内容？",
        "达摩祖师的'直指人心，见性成佛'是什么意思？",
        "达摩祖师传授了什么法门？",
        "达摩祖师的《二入四行论》讲什么？",
        "达摩祖师是如何选择传法弟子的？",
    ]},
    # 慧可大师 (二祖) - 6 questions
    {"figure": "慧可大师", "questions": [
        "慧可大师是谁？请介绍他的生平。",
        "慧可大师为什么断臂求法？",
        "慧可大师如何继承达摩的禅法？",
        "慧可大师'觅心了不可得'的公案是什么？",
        "慧可大师对中国禅宗有什么贡献？",
        "慧可大师的修行有什么特点？",
    ]},
    # 僧璨大师 (三祖) - 5 questions
    {"figure": "僧璨大师", "questions": [
        "僧璨大师是谁？请介绍他的生平。",
        "僧璨大师的《信心铭》讲什么？",
        "《信心铭》中'至道无难，唯嫌拣择'是什么意思？",
        "僧璨大师如何传承禅法？",
        "僧璨大师的禅学思想有什么特色？",
    ]},
    # 道信大师 (四祖) - 5 questions
    {"figure": "道信大师", "questions": [
        "道信大师是谁？请介绍他的生平。",
        "道信大师创建了什么样的禅修制度？",
        "道信大师的'一行三昧'是什么法门？",
        "道信大师对禅宗的发展有什么贡献？",
        "道信大师如何教导弘忍？",
    ]},
    # 弘忍大师 (五祖) - 6 questions
    {"figure": "弘忍大师", "questions": [
        "弘忍大师是谁？请介绍他的生平。",
        "弘忍大师的东山法门是什么？",
        "弘忍大师如何选择法嗣？",
        "弘忍大师为什么将衣钵传给慧能而不是神秀？",
        "弘忍大师在黄梅创建了什么道场？",
        "弘忍大师对禅宗的发展有什么影响？",
    ]},
    # 慧能大师 (六祖) - 15+ questions (CRITICAL)
    {"figure": "慧能大师", "questions": [
        "禅宗六祖是谁？",
        "慧能大师的生平是怎样的？",
        "慧能大师是如何得到五祖弘忍的衣钵传承的？",
        "慧能大师的'菩提本无树'偈语是什么？有什么深意？",
        "慧能大师与神秀大师的偈语有什么不同？",
        "《六祖坛经》的核心思想是什么？",
        "慧能大师的顿悟思想是什么？",
        "慧能大师为什么被称为禅宗六祖？",
        "慧能大师在南方弘法的经历是怎样的？",
        "慧能大师的'不思善不思恶'是什么意思？",
        "慧能大师的'何期自性，本自清净'是什么意思？",
        "慧能大师如何看待坐禅？",
        "慧能大师的无念、无相、无住是什么？",
        "慧能大师对后世禅宗有什么影响？",
        "慧能大师的弟子有哪些著名人物？",
    ]},
    # 神秀大师 - 5 questions
    {"figure": "神秀大师", "questions": [
        "神秀大师是谁？请介绍他的生平。",
        "神秀大师的渐修思想是什么？",
        "神秀大师的'身是菩提树'偈语是什么意思？",
        "神秀大师创立了什么宗派？",
        "神秀大师与慧能大师的思想有什么区别？",
    ]},
    # 马祖道一 - 8 questions
    {"figure": "马祖道一", "questions": [
        "马祖道一禅师是谁？请介绍他的生平。",
        "马祖道一的'平常心是道'是什么意思？",
        "马祖道一的'即心即佛'是什么思想？",
        "马祖道一禅师如何教化弟子？",
        "马祖道一磨砖作镜的公案是什么？",
        "马祖道一对禅宗的发展有什么贡献？",
        "马祖道一的禅法有什么特色？",
        "马祖道一与南岳怀让的关系是怎样的？",
    ]},
    # 百丈怀海 - 6 questions
    {"figure": "百丈怀海", "questions": [
        "百丈怀海禅师是谁？请介绍他的生平。",
        "百丈清规是什么？有什么意义？",
        "百丈怀海的'一日不作，一日不食'是什么意思？",
        "百丈怀海如何建立禅宗丛林制度？",
        "百丈怀海野狐禅的公案是什么？",
        "百丈怀海对禅宗的贡献有哪些？",
    ]},
    # 临济义玄 - 6 questions
    {"figure": "临济义玄", "questions": [
        "临济义玄禅师是谁？请介绍他的生平。",
        "临济宗的特色是什么？",
        "临济义玄的'四料拣'是什么？",
        "临济义玄的棒喝教学法是什么？",
        "临济义玄的'无位真人'是什么意思？",
        "临济宗在禅宗中的地位如何？",
    ]},
    # 洞山良价 - 5 questions
    {"figure": "洞山良价", "questions": [
        "洞山良价禅师是谁？请介绍他的生平。",
        "曹洞宗是怎么创立的？",
        "洞山良价的'五位君臣'是什么？",
        "洞山良价过水见影悟道的故事是怎样的？",
        "曹洞宗的修行特色是什么？",
    ]},
    # 玄奘法师 - 7 questions
    {"figure": "玄奘法师", "questions": [
        "玄奘法师是谁？请介绍他的生平。",
        "玄奘法师西行求法的经历是怎样的？",
        "玄奘法师翻译了哪些重要经典？",
        "玄奘法师创立了什么宗派？",
        "玄奘法师的《大唐西域记》记载了什么？",
        "玄奘法师的唯识学说是什么？",
        "玄奘法师对中国佛教有什么影响？",
    ]},
    # 鸠摩罗什 - 6 questions
    {"figure": "鸠摩罗什", "questions": [
        "鸠摩罗什是谁？请介绍他的生平。",
        "鸠摩罗什翻译了哪些重要经典？",
        "鸠摩罗什的翻译风格有什么特点？",
        "鸠摩罗什对中国佛教有什么贡献？",
        "鸠摩罗什翻译的《金刚经》有什么特点？",
        "鸠摩罗什与其他译经师相比有什么不同？",
    ]},
]

TERM_TOPICS = [
    # 基础概念
    {"category": "基础概念", "terms": [
        ("四圣谛", "佛教基本教义四圣谛"),
        ("苦谛", "四谛中的苦谛"),
        ("集谛", "四谛中的集谛"),
        ("灭谛", "四谛中的灭谛"),
        ("道谛", "四谛中的道谛"),
        ("八正道", "佛教修行八正道"),
        ("十二因缘", "佛教十二因缘"),
        ("无明", "十二因缘之无明"),
        ("五蕴", "五蕴皆空的五蕴"),
        ("色蕴", "五蕴之色蕴"),
        ("受蕴", "五蕴之受蕴"),
        ("想蕴", "五蕴之想蕴"),
        ("行蕴", "五蕴之行蕴"),
        ("识蕴", "五蕴之识蕴"),
        ("六度", "菩萨六度波罗蜜"),
        ("布施", "六度之布施波罗蜜"),
        ("持戒", "六度之持戒波罗蜜"),
        ("忍辱", "六度之忍辱波罗蜜"),
        ("精进", "六度之精进波罗蜜"),
        ("禅定", "六度之禅定波罗蜜"),
        ("般若", "六度之般若波罗蜜"),
        ("三法印", "佛教三法印"),
        ("缘起", "佛教缘起论"),
        ("因果", "佛教因果律"),
        ("轮回", "佛教六道轮回"),
        ("业力", "佛教业力思想"),
    ]},
    # 修行术语
    {"category": "修行术语", "terms": [
        ("涅槃", "佛教涅槃概念"),
        ("菩提", "佛教菩提觉悟"),
        ("三昧", "佛教三昧禅定"),
        ("止观", "天台止观法门"),
        ("念佛", "净土念佛法门"),
        ("参禅", "禅宗参禅方法"),
        ("观照", "佛教观照般若"),
        ("正念", "佛教正念修行"),
        ("四禅八定", "佛教四禅八定"),
        ("戒定慧", "佛教戒定慧三学"),
        ("三学", "戒定慧三学"),
        ("四念处", "四念处修行法"),
        ("七觉支", "三十七道品之七觉支"),
        ("三十七道品", "佛教三十七道品"),
        ("禅波罗蜜", "禅波罗蜜修行"),
        ("头陀行", "佛教头陀苦行"),
        ("经行", "佛教经行修行"),
        ("忏悔", "佛教忏悔法门"),
    ]},
    # 核心名相
    {"category": "核心名相", "terms": [
        ("空", "佛教空性思想"),
        ("中道", "佛教中道思想"),
        ("佛性", "佛教佛性论"),
        ("如来藏", "佛教如来藏思想"),
        ("真如", "佛教真如概念"),
        ("法身", "佛教三身之法身"),
        ("报身", "佛教三身之报身"),
        ("化身", "佛教三身之化身"),
        ("法界", "华严法界观"),
        ("实相", "佛教实相概念"),
        ("无我", "佛教无我思想"),
        ("无常", "佛教无常观"),
        ("慈悲", "佛教慈悲思想"),
        ("菩提心", "大乘菩提心"),
        ("回向", "佛教回向功德"),
        ("功德", "佛教功德概念"),
    ]},
    # 宗派名词
    {"category": "宗派", "terms": [
        ("禅宗", "中国禅宗"),
        ("净土宗", "中国净土宗"),
        ("天台宗", "中国天台宗"),
        ("华严宗", "中国华严宗"),
        ("唯识宗", "中国唯识宗"),
        ("密宗", "中国密宗"),
        ("中观", "佛教中观学派"),
        ("唯识", "佛教唯识学"),
        ("律宗", "中国律宗"),
        ("三论宗", "中国三论宗"),
    ]},
    # 果位名相
    {"category": "果位", "terms": [
        ("菩萨", "佛教菩萨概念"),
        ("阿罗汉", "佛教阿罗汉果"),
        ("声闻", "佛教声闻乘"),
        ("缘觉", "佛教缘觉乘"),
        ("佛", "佛教佛果"),
        ("辟支佛", "佛教辟支佛"),
        ("须陀洹", "佛教四果之初果"),
        ("斯陀含", "佛教四果之二果"),
        ("阿那含", "佛教四果之三果"),
        ("三乘", "佛教声闻缘觉菩萨三乘"),
        ("一佛乘", "法华经一佛乘思想"),
        ("十地", "菩萨十地"),
        ("等觉", "菩萨等觉位"),
        ("妙觉", "菩萨妙觉位"),
    ]},
]

SUTRA_TOPICS = [
    # 心经 - 6 questions
    {"sutra": "心经", "rag_query": "般若波罗蜜多心经", "questions": [
        "《心经》的核心要义是什么？",
        "《心经》中'色不异空，空不异色'是什么意思？",
        "《心经》中'照见五蕴皆空'是什么意思？",
        "《心经》为什么说'无苦集灭道'？",
        "《心经》的'揭谛揭谛'咒语是什么意思？",
        "《心经》在佛教中的地位如何？",
    ]},
    # 金刚经 - 6 questions
    {"sutra": "金刚经", "rag_query": "金刚般若波罗蜜经", "questions": [
        "《金刚经》的核心思想是什么？",
        "《金刚经》中'应无所住而生其心'是什么意思？",
        "《金刚经》中'凡所有相，皆是虚妄'是什么意思？",
        "《金刚经》的'四句偈'是什么？",
        "《金刚经》中'若以色见我，以音声求我，是人行邪道'是什么意思？",
        "《金刚经》为什么被禅宗特别推崇？",
    ]},
    # 法华经 - 5 questions
    {"sutra": "法华经", "rag_query": "妙法莲华经", "questions": [
        "《法华经》的核心思想是什么？",
        "《法华经》中的'方便品'讲什么？",
        "《法华经》中的火宅三车喻是什么意思？",
        "《法华经》的'一佛乘'思想是什么？",
        "《法华经》对中国佛教有什么影响？",
    ]},
    # 楞严经 - 5 questions
    {"sutra": "楞严经", "rag_query": "大佛顶首楞严经", "questions": [
        "《楞严经》的核心内容是什么？",
        "《楞严经》中的'七处征心'是什么？",
        "《楞严经》中的楞严咒有什么地位？",
        "《楞严经》对修行有什么指导意义？",
        "《楞严经》中的五十阴魔是什么？",
    ]},
    # 华严经 - 5 questions
    {"sutra": "华严经", "rag_query": "大方广佛华严经", "questions": [
        "《华严经》的核心思想是什么？",
        "《华严经》中的'法界缘起'是什么意思？",
        "《华严经》中善财童子五十三参的故事是什么？",
        "《华严经》的'一即一切，一切即一'是什么意思？",
        "《华严经》对华严宗的影响是什么？",
    ]},
    # 阿弥陀经 - 4 questions
    {"sutra": "阿弥陀经", "rag_query": "佛说阿弥陀经", "questions": [
        "《阿弥陀经》的核心内容是什么？",
        "《阿弥陀经》描述的西方极乐世界是怎样的？",
        "《阿弥陀经》中念佛往生的条件是什么？",
        "《阿弥陀经》在净土宗中的地位如何？",
    ]},
    # 无量寿经 - 4 questions
    {"sutra": "无量寿经", "rag_query": "佛说无量寿经", "questions": [
        "《无量寿经》的核心内容是什么？",
        "《无量寿经》中阿弥陀佛的四十八大愿是什么？",
        "《无量寿经》与《阿弥陀经》有什么异同？",
        "《无量寿经》对净土信仰有什么影响？",
    ]},
    # 地藏经 - 4 questions
    {"sutra": "地藏经", "rag_query": "地藏菩萨本愿经", "questions": [
        "《地藏经》的核心内容是什么？",
        "《地藏经》中地藏菩萨的大愿是什么？",
        "《地藏经》为什么被称为'佛门孝经'？",
        "《地藏经》对中国佛教信仰有什么影响？",
    ]},
    # 药师经 - 4 questions
    {"sutra": "药师经", "rag_query": "药师琉璃光如来本愿功德经", "questions": [
        "《药师经》的核心内容是什么？",
        "药师佛的十二大愿是什么？",
        "《药师经》在佛教修行中有什么作用？",
        "药师琉璃光如来的净土是怎样的？",
    ]},
    # 圆觉经 - 3 questions
    {"sutra": "圆觉经", "rag_query": "大方广圆觉修多罗了义经", "questions": [
        "《圆觉经》的核心思想是什么？",
        "《圆觉经》中的修行方法有哪些？",
        "《圆觉经》对禅宗有什么影响？",
    ]},
]

# ============================================================
# Generation Functions
# ============================================================

SYSTEM_PROMPT_PATRIARCH = """你是一位精通佛教历史和禅宗传承的学者。请根据提供的参考资料，准确回答关于佛教高僧大德的问题。

要求：
1. 回答必须基于参考资料中的事实，不可编造
2. 如果参考资料不足，可以补充公认的佛教史实，但不可虚构
3. 使用通俗易懂的语言，适当引用原文
4. 回答长度200-500字
5. 不要说"根据参考资料"等元语言，直接回答
6. 注意：慧能大师（638-713）是禅宗六祖，俗姓卢，新州（今广东新兴）人，此为确定事实"""

SYSTEM_PROMPT_TERM = """你是一位精通佛教义理的学者。请根据提供的参考资料，准确解释佛教名词术语。

要求：
1. 回答必须准确，基于佛教经典和权威辞典
2. 适当给出梵文/巴利文原词（如有）
3. 引用相关经典原文加以说明
4. 回答长度150-400字
5. 不要说"根据参考资料"等元语言，直接解释
6. 通俗易懂但保持学术准确性"""

SYSTEM_PROMPT_SUTRA = """你是一位精通佛教经典的学者。请根据提供的参考资料，准确回答关于佛教经典的问题。

要求：
1. 回答必须基于经典原文和权威注释
2. 适当引用经文原句
3. 解释要深入浅出，既有学理又通俗
4. 回答长度200-500字
5. 不要说"根据参考资料"等元语言，直接回答
6. 体现对经典的深入理解"""

async def generate_one(system_prompt: str, question: str, rag_query: str, rag_top_k: int = 8) -> Dict:
    """Generate one QA pair using RAG + LLM"""
    # Step 1: RAG search
    rag_results = await rag_search(rag_query, top_k=rag_top_k)
    context_parts = []
    for i, r in enumerate(rag_results):
        content = r["content"].strip()
        if content:
            src = r["metadata"].get("source", "")
            context_parts.append(f"[{i+1}] {src}: {content}")
    
    context = "\n\n".join(context_parts)
    
    # Step 2: LLM generation
    user_prompt = f"""参考资料：
{context}

问题：{question}

请根据参考资料准确回答上述问题。"""
    
    answer = await call_llm(system_prompt, user_prompt, temperature=0.7)
    
    return {
        "instruction": question,
        "output": answer.strip()
    }

async def generate_patriarch_qa() -> List[Dict]:
    """Generate 100 patriarch QA pairs"""
    results = []
    total_questions = sum(len(t["questions"]) for t in PATRIARCH_TOPICS)
    print(f"\n{'='*60}")
    print(f"生成禅宗祖师问答 (目标100条, 预设{total_questions}条)")
    print(f"{'='*60}")
    
    count = 0
    for topic in PATRIARCH_TOPICS:
        figure = topic["figure"]
        for q in topic["questions"]:
            count += 1
            rag_query = f"{figure} {q}"
            print(f"  [{count}/{total_questions}] {figure}: {q[:30]}...", end=" ", flush=True)
            
            try:
                result = await generate_one(SYSTEM_PROMPT_PATRIARCH, q, rag_query)
                results.append(result)
                print(f"✓ ({len(result['output'])}字)")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            # Rate limiting
            await asyncio.sleep(1.0)
    
    # If we need more to reach 100, generate additional
    if len(results) < 100:
        extra_questions = [
            ("慧能大师", "慧能大师在法性寺的经历是怎样的？"),
            ("慧能大师", "慧能大师如何看待文字与觉悟的关系？"),
            ("达摩大师", "达摩祖师的弟子有哪些？"),
            ("马祖道一", "马祖道一有哪些著名弟子？"),
            ("百丈怀海", "百丈怀海与马祖道一的关系是怎样的？"),
            ("临济义玄", "临济义玄与黄檗希运的关系是怎样的？"),
            ("玄奘法师", "玄奘法师在那烂陀寺的学习经历是怎样的？"),
            ("鸠摩罗什", "鸠摩罗什的译场是如何组织的？"),
            ("道信大师", "道信大师的修行生活是怎样的？"),
            ("弘忍大师", "弘忍大师如何选择继承人？"),
            ("慧能大师", "慧能大师圆寂后禅宗如何发展？"),
            ("慧能大师", "六祖坛经中记载了哪些重要法语？"),
        ]
        for figure, q in extra_questions:
            if len(results) >= 100:
                break
            count += 1
            rag_query = f"{figure} {q}"
            print(f"  [补充 {count}] {figure}: {q[:30]}...", end=" ", flush=True)
            try:
                result = await generate_one(SYSTEM_PROMPT_PATRIARCH, q, rag_query)
                results.append(result)
                print(f"✓ ({len(result['output'])}字)")
            except Exception as e:
                print(f"✗ Error: {e}")
            await asyncio.sleep(1.0)
    
    return results[:100]

async def generate_term_qa() -> List[Dict]:
    """Generate 100 term explanation QA pairs"""
    results = []
    all_terms = []
    for cat in TERM_TOPICS:
        for term, query in cat["terms"]:
            all_terms.append((cat["category"], term, query))
    
    total = len(all_terms)
    print(f"\n{'='*60}")
    print(f"生成佛学名词解释 (目标100条, 预设{total}条)")
    print(f"{'='*60}")
    
    for i, (category, term, query) in enumerate(all_terms):
        if len(results) >= 100:
            break
        
        question = f"什么是{term}？" if not term.startswith("佛") else f"{term}是什么？"
        # Vary question format
        if i % 5 == 1:
            question = f"请解释佛教中的{term}。"
        elif i % 5 == 2:
            question = f"佛教中的{term}是什么意思？"
        elif i % 5 == 3:
            question = f"{term}在佛教中指什么？"
        
        print(f"  [{i+1}/{min(total,100)}] [{category}] {term}: {question[:30]}...", end=" ", flush=True)
        
        try:
            result = await generate_one(SYSTEM_PROMPT_TERM, question, query)
            results.append(result)
            print(f"✓ ({len(result['output'])}字)")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        await asyncio.sleep(1.0)
    
    return results[:100]

async def generate_sutra_qa() -> List[Dict]:
    """Generate 50 sutra QA pairs"""
    results = []
    total_questions = sum(len(t["questions"]) for t in SUTRA_TOPICS)
    print(f"\n{'='*60}")
    print(f"生成经典问答 (目标50条, 预设{total_questions}条)")
    print(f"{'='*60}")
    
    count = 0
    for topic in SUTRA_TOPICS:
        sutra = topic["sutra"]
        base_query = topic["rag_query"]
        for q in topic["questions"]:
            if len(results) >= 50:
                break
            count += 1
            rag_query = f"{base_query} {q}"
            print(f"  [{count}/{min(total_questions,50)}] {sutra}: {q[:30]}...", end=" ", flush=True)
            
            try:
                result = await generate_one(SYSTEM_PROMPT_SUTRA, q, rag_query)
                results.append(result)
                print(f"✓ ({len(result['output'])}字)")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            await asyncio.sleep(1.0)
    
    return results[:50]

def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} items to {filepath}")

def verify_data(data: List[Dict], name: str):
    """Basic verification of generated data"""
    print(f"\n--- {name} 验证 ---")
    print(f"  总条数: {len(data)}")
    
    lengths = [len(d["output"]) for d in data]
    print(f"  回答长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    
    # Check for duplicates
    questions = [d["instruction"] for d in data]
    unique = set(questions)
    if len(unique) < len(questions):
        print(f"  ⚠ 发现 {len(questions) - len(unique)} 条重复问题！")
    else:
        print(f"  ✓ 无重复问题")
    
    # Check for key content in patriarch data
    if "祖师" in name:
        all_text = " ".join(d["output"] for d in data)
        key_checks = [
            ("慧能", "六祖"),
            ("达摩", "初祖"),
            ("慧可", "二祖"),
            ("马祖道一", "马祖"),
            ("百丈怀海", "百丈"),
            ("临济义玄", "临济"),
            ("玄奘", "玄奘"),
            ("鸠摩罗什", "鸠摩"),
        ]
        for name_check, label in key_checks:
            if name_check in all_text:
                print(f"  ✓ 包含{label}相关内容")
            else:
                print(f"  ✗ 缺少{label}相关内容！")

# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="v4.1 Training Data Generator")
    parser.add_argument("--type", choices=["patriarch", "term", "sutra", "all"], default="all",
                       help="Type of data to generate")
    parser.add_argument("--dry-run", action="store_true", help="Test with 2 items per category")
    args = parser.parse_args()
    
    print("=" * 60)
    print("善知识 v4.1 Training Data Generator")
    print("=" * 60)
    print(f"  RAG: Qdrant @ {QDRANT_HOST}:{QDRANT_PORT}, collection={COLLECTION_NAME}")
    print(f"  LLM: {LLM_MODEL} via DashScope")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'FULL GENERATION'}")
    
    # Test connections
    print("\n--- 连接测试 ---")
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"  ✓ Qdrant: {info.points_count} points")
    except Exception as e:
        print(f"  ✗ Qdrant: {e}")
        return
    
    try:
        test_vec = await get_embedding("测试")
        print(f"  ✓ Embedding: dim={len(test_vec)}")
    except Exception as e:
        print(f"  ✗ Embedding: {e}")
        return
    
    try:
        test_answer = await call_llm("你是助手", "说'连接成功'", temperature=0.1)
        print(f"  ✓ LLM: {test_answer[:20]}...")
    except Exception as e:
        print(f"  ✗ LLM: {e}")
        return
    
    start_time = time.time()
    
    if args.type in ["patriarch", "all"]:
        if args.dry_run:
            # Only test with first 2
            for t in PATRIARCH_TOPICS:
                t["questions"] = t["questions"][:1]
        patriarch_data = await generate_patriarch_qa()
        save_jsonl(patriarch_data, OUTPUT_DIR / "patriarch_qa.jsonl")
        verify_data(patriarch_data, "禅宗祖师问答")
    
    if args.type in ["term", "all"]:
        if args.dry_run:
            for cat in TERM_TOPICS:
                cat["terms"] = cat["terms"][:2]
        term_data = await generate_term_qa()
        save_jsonl(term_data, OUTPUT_DIR / "term_qa.jsonl")
        verify_data(term_data, "佛学名词解释")
    
    if args.type in ["sutra", "all"]:
        if args.dry_run:
            for t in SUTRA_TOPICS:
                t["questions"] = t["questions"][:1]
        sutra_data = await generate_sutra_qa()
        save_jsonl(sutra_data, OUTPUT_DIR / "sutra_qa.jsonl")
        verify_data(sutra_data, "经典问答")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"生成完成！耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    
    # Final summary
    for fname in ["patriarch_qa.jsonl", "term_qa.jsonl", "sutra_qa.jsonl"]:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                lines = f.readlines()
            print(f"  {fname}: {len(lines)} 条")

if __name__ == "__main__":
    asyncio.run(main())
