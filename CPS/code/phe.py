# phe.py
import re, json
from typing import List, Dict, Tuple
from config import embed
from ckg import ckg

def llm_api(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """占位；替换成 OpenAI/Baichuan/DeepSeek 等真实接口"""
    return (
        "Learning objectives: (i) understand Snell's law qualitatively; "
        "(ii) apply to daily phenomena. "
        "Common misconceptions: bending-vs-speed. "
        "Core activities: laser-water-bottle refraction test. "
        "Required kits: laser pointer ≤$1, clear bottle. "
        "5E timing: Engage 5min, Explore 15min, Explain 10min, Elaborate 5min, Evaluate 5min."
    )

def phe(raw_query: str) -> Dict:
    """返回结构化假设 + 片段"""
    system = ("You are an expert curriculum designer. Generate a concise 5-section hypothesis: "
              "(i) learning objectives, (ii) common misconceptions, (iii) core activities, "
              "(iv) required lab kits, (v) 5E-stage timing. Inline constraints: ≤40min, low-cost.")
    prompt = f"{system}\nTeacher Query: {raw_query}"
    ho_text = llm_api(prompt, max_tokens=400)

    # 简易正则抽取
    fragments = []
    obj = re.findall(r"objectives?:[\s\-\(i\)](.+?)(?=\s*(miscon|core|required|5E|timing|\Z))", ho_text, re.I | re.S)
    if obj: fragments.append(("LearningObjective", obj[0][0].strip()))
    mis = re.findall(r"misconceptions?:[\s\-\(i\)](.+?)(?=\s*(core|required|5E|timing|\Z))", ho_text, re.I | re.S)
    if mis: fragments.append(("Misconception", mis[0][0].strip()))
    act = re.findall(r"activities?:[\s\-\(i\)](.+?)(?=\s*(required|kits|5E|timing|\Z))", ho_text, re.I | re.S)
    if act: fragments.append(("Activity", act[0][0].strip()))
    kit = re.findall(r"kits?:[\s\-\(i\)](.+?)(?=\s*(5E|timing|\Z))", ho_text, re.I | re.S)
    if kit: fragments.append(("Resource", kit[0][0].strip()))
    tim = re.findall(r"timing:?[\s\-\(i\)](.+?)(?=\Z)", ho_text, re.I | re.S)
    if tim: fragments.append(("Time", tim[0].strip()))

    return {"hypothesis": ho_text, "fragments": fragments}