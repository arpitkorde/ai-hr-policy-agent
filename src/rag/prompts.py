"""Versioned prompt templates for the HR Policy Agent.

Centralized prompt management with versioning for A/B testing
and iterative improvement of RAG response quality.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate

# --- Prompt Version Registry ---
PROMPT_VERSIONS = {}


def register_prompt(version: str):
    """Decorator to register a prompt version."""
    def decorator(func):
        PROMPT_VERSIONS[version] = func
        return func
    return decorator


# ============================================================
# Version 1.0 — Baseline HR Policy QA Prompt
# ============================================================
@register_prompt("v1.0")
def prompt_v1():
    """Baseline prompt: direct Q&A with source citation."""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI HR Policy Assistant. Your role is to answer employee
questions accurately based ONLY on the provided company policy documents.

RULES:
1. Answer ONLY based on the provided context. Do NOT make up information.
2. If the answer is not in the context, say: "I don't have information about
   this in our current policy documents. Please contact HR directly."
3. Always cite which document your answer comes from.
4. Be professional, empathetic, and clear.
5. If a policy has specific conditions or exceptions, mention them.

CONTEXT FROM POLICY DOCUMENTS:
{context}""",
        ),
        ("human", "{question}"),
    ])


# ============================================================
# Version 2.0 — Enhanced with structured output & confidence
# ============================================================
@register_prompt("v2.0")
def prompt_v2():
    """Enhanced prompt: structured output with confidence scoring."""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI HR Policy Assistant for answering employee questions
about company policies. You must ONLY use the provided context documents.

INSTRUCTIONS:
1. Read the context carefully and identify relevant policy sections.
2. Provide a clear, structured answer.
3. Rate your confidence: HIGH (directly stated in policy), MEDIUM (inferred
   from policy), or LOW (not clearly covered).
4. Always cite the source document and section.
5. If the context doesn't contain the answer, clearly state that and
   suggest contacting HR.

FORMAT YOUR RESPONSE AS:
**Answer:** [Your answer here]

**Source:** [Document name, page/section if available]

**Confidence:** [HIGH / MEDIUM / LOW]

CONTEXT FROM POLICY DOCUMENTS:
{context}""",
        ),
        ("human", "{question}"),
    ])


# ============================================================
# Standalone QA prompt for non-chat usage
# ============================================================
STANDALONE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context from HR policy documents to
answer the employee's question. If you don't know the answer based on
the context, say you don't know. Always cite your sources.

Context:
{context}

Question: {question}

Answer:""",
)


def get_prompt(version: str = "v2.0") -> ChatPromptTemplate:
    """Get a prompt template by version.

    Args:
        version: The prompt version string (e.g., "v1.0", "v2.0").

    Returns:
        ChatPromptTemplate for the specified version.

    Raises:
        KeyError: If the version is not registered.
    """
    if version not in PROMPT_VERSIONS:
        available = ", ".join(PROMPT_VERSIONS.keys())
        raise KeyError(
            f"Prompt version '{version}' not found. Available: {available}"
        )
    return PROMPT_VERSIONS[version]()


def list_prompt_versions() -> list[str]:
    """List all registered prompt versions."""
    return list(PROMPT_VERSIONS.keys())
