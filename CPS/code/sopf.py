# sopf.py
import numpy as np
from sklearn.neighbors import KernelDensity
from typing import List, Dict, Tuple
from config import embed

def sopf(chains: List[Dict], fragments: List[Tuple[str, str]], top_k: int = 12) -> List[Dict]:
    """对比式核密度重排序 + 超参自由切割（gap statistic）"""
    f_vecs = np.stack([embed(f"{t} {txt}") for t, txt in fragments])
    c_vecs = np.stack([embed(ch["chain"]) for ch in chains])

    # 冗余密度 ρ
    bandwidth = np.median([np.linalg.norm(c_vecs[i] - c_vecs[j]) for i in range(len(chains)) for j in range(i + 1, len(chains))])
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(c_vecs)
    log_dens = kde.score_samples(c_vecs)
    rho = np.exp(log_dens)

    # 覆盖梯度 γ
    gamma = np.zeros(len(chains))
    for i, cv in enumerate(c_vecs):
        gap_sum = 0
        for f_vec in f_vecs:
            gap_sum += max(0, 1 - np.dot(cv, f_vec)) * np.dot(cv, f_vec)
        gamma[i] = gap_sum

    # 对比得分 & 自动切割
    scores = gamma / (rho + 1e-8)
    order = np.argsort(-scores)
    for i in range(1, len(order)):
        if (scores[order[i - 1]] - scores[order[i]]) / scores[order[i - 1]] > 0.05:
            cut = i
            break
    else:
        cut = min(top_k, len(order))
    return [chains[order[i]] for i in range(cut)]