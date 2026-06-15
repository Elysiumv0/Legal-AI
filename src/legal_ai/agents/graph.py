from langgraph.graph import StateGraph, END
from legal_ai.agents.state import AgentState
from legal_ai.agents.nodes.researcher import ResearcherNode
from legal_ai.agents.nodes.critic import CriticNode
from legal_ai.agents.nodes.drafter import DrafterNode
from legal_ai.agents.nodes.reviewer import ReviewerNode

from legal_ai.retrieval.sparse.bm25 import BM25Retriever
from legal_ai.retrieval.dense.retriever import DenseRetriever, VectorStore
from legal_ai.models.embeddings import EmbeddingModel
from legal_ai.retrieval.hybrid.fusion import HybridRetriever
from legal_ai.models.reranker import Reranker
from legal_ai.models.llm import LLMClient

def route_after_critic(state: AgentState) -> str:
    if state.get("context_sufficient") or state.get("retrieval_attempts", 0) >= 3:
        return "drafter"
    return "researcher"

def build_graph(
    researcher: ResearcherNode,
    critic: CriticNode,
    drafter: DrafterNode,
    reviewer: ReviewerNode,
):
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("researcher", researcher.run)
    graph.add_node("critic", critic.run)
    graph.add_node("drafter", drafter.run)
    graph.add_node("reviewer", reviewer.run)
    
    # Entry point
    graph.set_entry_point("researcher")
    
    # Edges
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda state: "drafter" if state["context_sufficient"] else "researcher",
        {"researcher": "researcher", "drafter": "drafter"}
    )
    graph.add_edge("drafter", "reviewer")
    graph.add_edge("reviewer", END)
    
    return graph.compile()

def build_agent(config: dict):
    bm25 = BM25Retriever()
    bm25.load(config["bm25_path"])
    embedder = EmbeddingModel(config["embedding_model"])
    store    = VectorStore(config["qdrant_path"])
    dense    = DenseRetriever(store, embedder)
    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker(config["reranker_model"])
    llm = LLMClient(config["llm_model_name"])
    researcher = ResearcherNode(hybrid, reranker, llm)
    critic     = CriticNode(llm)
    drafter    = DrafterNode(llm)
    reviewer   = ReviewerNode(llm)
    return build_graph(researcher, critic, drafter, reviewer)

def run_agent(graph, question: str, question_id: int = 0) -> dict:
    initial_state = {
        "question":            question,
        "question_id":         question_id,
        "sub_queries":         [],
        "retrieved_chunks":    [],
        "retrieval_attempts":  0,
        "context_sufficient":  False,
        "critic_feedback":     "",
        "draft_answer":        "",
        "final_answer":        "",
        "relevant_docs":       [],
        "relevant_articles":   [],
        "messages":            [],
    }
    final_state = graph.invoke(initial_state)
    return {
        "id":               question_id,
        "question":         question,
        "answer":           final_state["final_answer"],
        "relevant_docs":    final_state["relevant_docs"],
        "relevant_articles": final_state["relevant_articles"],
    }
