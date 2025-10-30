# hydra-core/fraud/graph_engine.py
"""
Risk Graph (MVP, in-memory)
---------------------------
Purpose:
  • Maintain a lightweight, auditable co-occurrence graph of entities and risk facts
    to provide AFPS with simple, network-aware hints (links & clusters).
  • Zero external dependencies. Designed as a drop-in adapter you can later swap
    for Neo4j/TigerGraph/AWS Neptune with the same public API.

Concepts:
  • Node: string id (loan:LN-123, borrower:B-42, email:hash, addr:hash, device:hash, flag:income_gap)
  • Edge: undirected (u, v) with integer weight (co-occurrence count)
  • Snapshot: stable JSON of nodes/edges + summary stats + governance hash

Use-cases:
  • “This loan shares device/email with 3 other high-AFPS loans in last 30d.”
  • “income_gap co-occurs with id_mismatch frequently for this borrower cluster.”
"""

from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, Set, List
from collections import defaultdict, Counter, OrderedDict
import time

from utils.utils import blake2b_hex, canonical_json, clamp

GRAPH_SCHEMA_VERSION = "risk-graph-v0"
POLICY_ID = "graph-policy-2025-10-a"

# ─────────────────────────────────────────────────────────────────────
# Public types (simple dicts for MVP)
# ─────────────────────────────────────────────────────────────────────

class RiskGraph:
    """
    Minimal in-memory graph.
    Not thread-safe by design (FastAPI single worker ok for MVP).
    If needed, wrap calls with a lock in the DB adapter later.
    """

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[Tuple[str, str], int] = defaultdict(int)
        self._tags: Dict[str, Set[str]] = defaultdict(set)   # node_id -> {tag,...}
        self._seen: Counter = Counter()                      # node_id -> seen count
        self._ts_created = int(time.time())

    # ── CRUD-ish ────────────────────────────────────────────────────

    def upsert_nodes(self, node_ids: Iterable[str], tags: Dict[str, Iterable[str]] | None = None) -> None:
        for n in node_ids:
            if not n:
                continue
            self._nodes.add(n)
            self._seen[n] += 1
        if tags:
            for n, tg in tags.items():
                if n in self._nodes:
                    self._tags[n].update([str(t).lower() for t in tg])

    def add_edge(self, a: str, b: str, weight: int = 1) -> None:
        if not a or not b or a == b:
            return
        if a not in self._nodes or b not in self._nodes:
            # auto-upsert nodes
            self.upsert_nodes([a, b])
        u, v = sorted([a, b])
        self._edges[(u, v)] += int(max(weight, 1))

    def add_cooccurrence(self, nodes: Iterable[str]) -> None:
        arr = [n for n in nodes if n]
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                self.add_edge(arr[i], arr[j], 1)

    # ── Queries ─────────────────────────────────────────────────────

    def degree(self, n: str) -> int:
        if n not in self._nodes:
            return 0
        d = 0
        for (u, v), w in self._edges.items():
            if u == n or v == n:
                d += w
        return d

    def neighbors(self, n: str, k: int = 10) -> List[Tuple[str, int]]:
        nbrs: Dict[str, int] = defaultdict(int)
        for (u, v), w in self._edges.items():
            if u == n:
                nbrs[v] += w
            elif v == n:
                nbrs[u] += w
        return sorted(nbrs.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def cooccurs_with(self, a: str, b: str) -> int:
        u, v = sorted([a, b])
        return self._edges.get((u, v), 0)

    def top_links(self, k: int = 10) -> List[Tuple[str, str, int]]:
        pairs = [ (u, v, w) for (u, v), w in self._edges.items() ]
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:k]

    def tag(self, n: str) -> Set[str]:
        return set(self._tags.get(n, set()))

    # ── AFPS-oriented helpers ───────────────────────────────────────

    def shortlinks_for_loan(self, loan_id: str, context_nodes: Iterable[str], max_links: int = 5) -> List[Dict[str, Any]]:
        """
        Given a loan and a set of context nodes (borrower/email/device/address/flags),
        return the strongest external links (who/what it connects to) with hints.
        """
        hints: List[Dict[str, Any]] = []
        context = [c for c in context_nodes if c]
        seen_pairs: Set[Tuple[str, str]] = set()

        for c in context:
            for nbr, w in self.neighbors(c, k=10):
                if nbr.startswith("loan:") and nbr != loan_id:
                    pair = tuple(sorted([loan_id, nbr]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # band heuristic by edge weight
                    band = "LOW"
                    if w >= 5: band = "HIGH"
                    elif w >= 2: band = "MED"

                    hints.append({
                        "link_to": nbr,
                        "via": c,
                        "strength": w,
                        "band": band,
                        "notes": self._mk_hint_note(c, nbr, w),
                    })
        hints.sort(key=lambda h: h["strength"], reverse=True)
        return hints[:max_links]

    def cluster_score(self, node_ids: Iterable[str]) -> float:
        """
        Simple density metric 0..1: edges / max_possible_edges among the set.
        """
        nodes = [n for n in node_ids if n in self._nodes]
        n = len(nodes)
        if n < 2:
            return 0.0
        edges_present = 0
        for i in range(n):
            for j in range(i + 1, n):
                u, v = sorted([nodes[i], nodes[j]])
                if self._edges.get((u, v), 0) > 0:
                    edges_present += 1
        max_edges = n * (n - 1) // 2
        return round(edges_present / max_edges, 4)

    # ── Snapshots & Governance ──────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        Stable snapshot for audit/telemetry. Do not include volatile timestamps
        in the hashed payload; keep them adjacent.
        """
        payload = OrderedDict()
        payload["schema_version"] = GRAPH_SCHEMA_VERSION
        payload["policy_id"] = POLICY_ID
        payload["node_count"] = len(self._nodes)
        payload["edge_count"] = len(self._edges)
        payload["top_links"] = [
            {"u": u, "v": v, "w": w} for (u, v, w) in self.top_links(k=20)
        ]
        payload["sample_nodes"] = sorted(list(self._sample_nodes(12)))
        payload["tags"] = {n: sorted(list(ts)) for n, ts in list(self._tags.items())[:20]}

        hashable = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "policy_id": POLICY_ID,
            "node_count": payload["node_count"],
            "edge_count": payload["edge_count"],
            "top_links": [(t["u"], t["v"], t["w"]) for t in payload["top_links"]],
            "sample_nodes": payload["sample_nodes"],
        }
        return {
            "snapshot": payload,
            "governance": {
                "graph_hash": blake2b_hex(canonical_json(hashable)),
                "ts_created": self._ts_created,
                "ts_snapshot": int(time.time()),
            },
        }

    # ── Internals ───────────────────────────────────────────────────

    def _mk_hint_note(self, via: str, loan: str, w: int) -> str:
        if via.startswith("device:"):
            return f"Shared device across {w} interactions → {loan}"
        if via.startswith("email:"):
            return f"Shared email fingerprint ({w} links) → {loan}"
        if via.startswith("addr:"):
            return f"Shared address cluster ({w}) → {loan}"
        if via.startswith("flag:"):
            return f"Risk flag co-occurred {w}× → {loan}"
        return f"Co-occurrence strength {w} via {via}"

    def _sample_nodes(self, k: int) -> Set[str]:
        # Prefer more-connected nodes for sampling; fall back to arbitrary.
        if not self._nodes:
            return set()
        by_deg = sorted(self._nodes, key=lambda n: self.degree(n), reverse=True)
        return set(by_deg[:k])


# ─────────────────────────────────────────────────────────────────────
# Convenience functions (adapter surface for AFPS & feature builder)
# ─────────────────────────────────────────────────────────────────────

_GLOBAL_GRAPH: RiskGraph | None = None

def get_graph() -> RiskGraph:
    global _GLOBAL_GRAPH
    if _GLOBAL_GRAPH is None:
        _GLOBAL_GRAPH = RiskGraph()
    return _GLOBAL_GRAPH

def ingest_event_links(
    loan_id: str,
    borrower_id: str | None = None,
    email_fp: str | None = None,
    addr_fp: str | None = None,
    device_fp: str | None = None,
    risk_flags: Iterable[str] | None = None,
) -> None:
    """
    Add co-occurrences for this event. Normalize ids to namespaced tokens.
    """
    g = get_graph()
    nodes = [f"loan:{loan_id}"]
    tags = {f"loan:{loan_id}": ["loan"]}

    if borrower_id:
        nodes.append(f"borrower:{borrower_id}")
        tags[f"borrower:{borrower_id}"] = ["borrower"]
    if email_fp:
        nodes.append(f"email:{email_fp}")
        tags[f"email:{email_fp}"] = ["email"]
    if addr_fp:
        nodes.append(f"addr:{addr_fp}")
        tags[f"addr:{addr_fp}"] = ["address"]
    if device_fp:
        nodes.append(f"device:{device_fp}")
        tags[f"device:{device_fp}"] = ["device"]
    if risk_flags:
        for rf in risk_flags:
            k = f"flag:{str(rf).lower()}"
            nodes.append(k)
            tags[k] = ["flag"]

    g.upsert_nodes(nodes, tags)
    g.add_cooccurrence(nodes)

def hints_for_loan(
    loan_id: str,
    borrower_id: str | None,
    email_fp: str | None,
    addr_fp: str | None,
    device_fp: str | None,
    risk_flags: Iterable[str] | None = None,
    max_links: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return strongest network hints for AFPS narratives.
    """
    g = get_graph()
    context = [
        f"borrower:{borrower_id}" if borrower_id else None,
        f"email:{email_fp}" if email_fp else None,
        f"addr:{addr_fp}" if addr_fp else None,
        f"device:{device_fp}" if device_fp else None,
    ]
    if risk_flags:
        context.extend([f"flag:{str(rf).lower()}" for rf in risk_flags])
    return g.shortlinks_for_loan(f"loan:{loan_id}", context_nodes=context, max_links=max_links)


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    g = get_graph()
    ingest_event_links(
        loan_id="LN-1",
        borrower_id="B-1",
        email_fp="e:abc",
        addr_fp="a:xyz",
        device_fp="d:777",
        risk_flags=["income_gap", "doc_missing"],
    )
    ingest_event_links(
        loan_id="LN-2",
        borrower_id="B-9",
        email_fp="e:abc",
        addr_fp="a:xyz",
        device_fp="d:777",
        risk_flags=["income_gap"],
    )
    print("Neighbors(email:e:abc):", g.neighbors("email:e:abc"))
    print("Hints for LN-2:",
          hints_for_loan("LN-2", "B-9", "e:abc", "a:xyz", "d:777", ["income_gap"]))
    snap = g.snapshot()
    print("Snapshot:", snap["governance"])
