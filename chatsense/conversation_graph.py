"""
Conversation Interaction Graph Analysis Module for ContextGuard AI.
Models conversational turns and speaker exchanges as directed graphs using NetworkX.
Generates interaction topology, turn-taking reciprocity metrics, and anonymized graph diagrams.

DISCLAIMER:
Graph metrics (e.g., node degree, turn ratios) describe communication volume and flow.
They MUST NOT be interpreted as clinical proof of dominance, guilt, or interpersonal abuse.
"""
import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ConversationGraphAnalyzer:
    """
    Constructs and analyzes directed conversation graphs from WhatsApp chat logs.
    """

    def __init__(self, output_dir: str = "static/images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Builds a directed conversation graph:
        - Participant nodes and Message turn nodes
        - Temporal adjacency edges (M_i -> M_{i+1})
        - Interaction edges (User_A -> User_B)
        """
        G = nx.DiGraph()
        if df.empty or "sender" not in df.columns:
            return G

        # Add participant nodes
        senders = df["sender"].unique().tolist()
        for s in senders:
            G.add_node(f"User:{s}", type="participant", label=s)

        # Track turn transitions between senders
        prev_sender = None
        records = df.to_dict('records')
        for i, row in enumerate(records):
            curr_sender = row.get("sender", "Unknown")
            msg_id = f"M{i}"

            # Add message node with relevant linguistic attributes
            G.add_node(
                msg_id,
                type="message",
                index=i,
                sender=curr_sender,
                sentiment=row.get("sentiment", "Neutral"),
                raw_score=row.get("raw_score", 0.0),
                dominant_emotion=row.get("dominant_emotion", "neutral"),
                top_pattern=row.get("top_pattern", "none"),
                is_reply=row.get("is_reply", False),
            )

            # Edge from sender to message
            G.add_edge(f"User:{curr_sender}", msg_id, relationship="authored")

            # Sequential temporal flow edge
            if i > 0:
                G.add_edge(f"M{i-1}", msg_id, relationship="temporal_sequence")

            # Cross-speaker conversational response edge
            if prev_sender and prev_sender != curr_sender:
                u_from = f"User:{prev_sender}"
                u_to = f"User:{curr_sender}"
                if G.has_edge(u_from, u_to):
                    G[u_from][u_to]["weight"] += 1
                else:
                    G.add_edge(u_from, u_to, relationship="turn_transition", weight=1)

            prev_sender = curr_sender

        return G

    def compute_graph_metrics(self, df: pd.DataFrame, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Computes interaction flow, reciprocity, and turn-taking distribution.
        """
        if G.number_of_nodes() == 0 or df.empty:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "participant_nodes": 0,
                "reciprocity_ratio": 0.0,
                "conversation_density": 0.0,
                "turn_transitions": 0,
                "turn_transitions_count": 0,
                "top_initiator": "N/A",
                "reciprocity_interpretation": "No multi-turn interaction detected.",
                "interaction_matrix": [],
                "disclaimer": (
                    "Graph metrics describe statistical communication flow only and "
                    "do not evaluate personality or relationship power dynamics."
                ),
            }

        participants = [n for n, d in G.nodes(data=True) if d.get("type") == "participant"]

        # Calculate reciprocity between participants
        transitions = 0
        interaction_pairs = []
        for u1 in participants:
            for u2 in participants:
                if u1 != u2 and G.has_edge(u1, u2):
                    w = G[u1][u2].get("weight", 1)
                    transitions += w
                    interaction_pairs.append({
                        "from": G.nodes[u1].get("label", u1),
                        "to": G.nodes[u2].get("label", u2),
                        "count": w,
                    })

        # Reciprocity calculation
        p_subgraph = G.subgraph(participants)
        try:
            reciprocity = round(float(nx.reciprocity(p_subgraph)), 3)
        except Exception:
            reciprocity = 0.0

        try:
            density = round(float(nx.density(p_subgraph)), 3)
        except Exception:
            density = 0.0

        # Top initiator from df
        top_initiator = "N/A"
        if not df.empty and "sender" in df.columns:
            top_initiator = str(df["sender"].value_counts().index[0])

        interp = (
            "Mutual, balanced conversational reciprocity observed."
            if reciprocity >= 0.6
            else "Asymmetrical response distribution observed across participants."
        )

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "participant_nodes": len(participants),
            "reciprocity_ratio": reciprocity,
            "conversation_density": density,
            "turn_transitions": transitions,
            "turn_transitions_count": transitions,
            "top_initiator": top_initiator,
            "reciprocity_interpretation": interp,
            "interaction_matrix": interaction_pairs,
            "disclaimer": (
                "Notice: Node counts and reciprocity reflect turn-taking frequency. "
                "They must not be taken as proof of conversational dominance or psychological traits."
            ),
        }

    def generate_graph_chart(self, df: pd.DataFrame, filename: str = "conversation_graph.png") -> Optional[str]:
        """
        Generates and saves a clean NetworkX visualization of participant communication flows.
        """
        if df.empty or "sender" not in df.columns:
            return None

        # Build collapsed participant interaction graph
        P = nx.DiGraph()
        sender_counts = df["sender"].value_counts()
        # For group chats with dozens/hundreds of participants, focus visualization on top 12 active speakers
        if len(sender_counts) > 12:
            top_senders = set(sender_counts.head(12).index)
        else:
            top_senders = set(sender_counts.index)

        for s in top_senders:
            P.add_node(s, count=int(sender_counts[s]))

        prev_s = None
        for s in df["sender"]:
            if s in top_senders:
                if prev_s and prev_s != s and prev_s in top_senders:
                    if P.has_edge(prev_s, s):
                        P[prev_s][s]["weight"] += 1
                    else:
                        P.add_edge(prev_s, s, weight=1)
                prev_s = s
            else:
                prev_s = None

        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        pos = nx.spring_layout(P, seed=42, k=1.5)
        node_sizes = [min(3000, max(1200, P.nodes[n].get("count", 1) * 35)) for n in P.nodes()]

        nx.draw_networkx_nodes(
            P, pos, ax=ax, node_color="#3b82f6", node_size=node_sizes, alpha=0.9
        )
        nx.draw_networkx_labels(
            P, pos, ax=ax, font_color="#ffffff", font_family="sans-serif", font_weight="bold", font_size=11
        )

        edges = P.edges(data=True)
        if edges:
            weights = [min(6, max(1, e[2].get("weight", 1) * 0.4)) for e in edges]
            nx.draw_networkx_edges(
                P, pos, ax=ax, edge_color="#60a5fa", width=weights,
                arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.15"
            )

        ax.set_title("Participant Interaction & Turn-Taking Network", color="#e2e8f0", fontsize=12, pad=12)
        plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Return relative web path
        rel_path = os.path.relpath(out_path, start="static").replace("\\", "/")
        return rel_path

    def export_anonymized_summary(self, df: pd.DataFrame, G: nx.DiGraph, out_filepath: str) -> None:
        """Exports graph statistics without any raw participant names or text."""
        metrics = self.compute_graph_metrics(df, G)
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
