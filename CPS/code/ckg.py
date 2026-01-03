# ckg.py
import networkx as nx
from config import embed

class CKG:
    """极简 CKG；生产环境可替换为从国标/教材解析的实体与关系"""
    def __init__(self):
        self.G = nx.DiGraph()
        self._build()

    def _build(self):
        nodes = {
            "light-source":           {"type": "Concept", "desc": "光源，直线传播前提"},
            "rectilinear-propagation":{"type": "Concept", "desc": "光在同种均匀介质中直线传播"},
            "refraction":             {"type": "Concept", "desc": "光从一种介质斜射入另一种介质时的偏折"},
            "bending-vs-speed":       {"type": "Misconception", "desc": "学生常误认为弯曲是因为速度变慢"},
            "laser-water-bottle":     {"type": "Activity", "desc": "激光笔+水瓶观察折射"},
            "water-bottle-laser-kit": {"type": "Resource", "desc": "低成本水瓶激光实验套件，成本≤$1"},
        }
        edges = [
            ("light-source", "rectilinear-propagation", "prerequisite-of"),
            ("rectilinear-propagation", "refraction", "prerequisite-of"),
            ("bending-vs-speed", "refraction", "addresses-misconception"),
            ("laser-water-bottle", "water-bottle-laser-kit", "uses-kit"),
            ("laser-water-bottle", "refraction", "addresses-misconception"),
        ]
        for n, attr in nodes.items():
            self.G.add_node(n, **attr)
        for u, v, r in edges:
            self.G.add_edge(u, v, relation=r)

        # 预计算句向量
        for n in self.G.nodes:
            self.G.nodes[n]["vec"] = embed(f"{self.G.nodes[n]['type']} {n} {self.G.nodes[n]['desc']}")

    def nodes(self, data=False):
        return self.G.nodes(data=data)

    def successors(self, n):
        return self.G.successors(n)

    def shortest_path(self, u, v):
        return nx.shortest_path(self.G, u, v)

# 全局单例
ckg = CKG()