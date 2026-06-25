from typing import TypedDict, Annotated, Any
import operator

class AgentState(TypedDict):
    question:       str             
    question_id:    int | None      

    # ── NER entities từ enriched data ──
    entities:       dict[str, Any]   # law_ids, legal_actions, legal_objects, penalties, primary_law_id, ...

    sub_queries:    list[str]        
    retrieved_chunks: list[dict]    
    retrieval_attempts: int         

    context_sufficient: bool         
    critic_feedback:    str         

    draft_answer:   str          
    final_answer:   str             

    relevant_docs:     list[str]    
    relevant_articles: list[str]     

    messages: Annotated[list, operator.add]