# main.py
import json
from phe import phe
from sopf import sopf
from ckg import ckg

def llm_reader(raw_query: str, rc_star: list) -> str:
    appendix = "\n".join([f"[{ch['chain_type']}] {ch['chain']}" for ch in rc_star])
    prompt = (f"Using the attached pedagogical fragments, write a lesson plan for the query:\n{raw_query}\n"
              f"Fragments:\n{appendix}\n"
              "Output JSON with fields: lesson_title, learning_objectives, teaching_sequence (5E), total_cost_usd.")
    # 占位，同 phe.llm_api
    from phe import llm_api
    return llm_api(prompt, max_tokens=600)

def run_cps(raw_query: str):
    # 1. PHE
    phe_out = phe(raw_query)
    # 2. CGAO
    from phe import retrieve_chains     # 为了导入方便，把 retrieve_chains 也放进 phe.py
    chains = retrieve_chains(phe_out["fragments"])
    # 3. SoPF
    rc_star = sopf(chains, phe_out["fragments"])
    # 4. Reader
    plan = llm_reader(raw_query, rc_star)
    return {"hypothesis": phe_out["hypothesis"], "rc_star": rc_star, "lesson_plan": plan}

if __name__ == "__main__":
    q = "Grade 8 light-refraction lesson, 40 min, 5E model, low-cost"
    print(json.dumps(run_cps(q), ensure_ascii=False, indent=2))