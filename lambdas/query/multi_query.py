"""
Multi-query comparison handler for cross-document intelligence.

Detects comparison questions, generates sub-queries for each entity,
retrieves from multiple perspectives, and synthesizes findings.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Comparison question patterns
COMPARISON_PATTERNS = [
    r"\bcompare\b.*\b(?:and|vs\.?|versus|to|with)\b",
    r"\bdifference(?:s)?\s+between\b",
    r"\b(?:vs\.?|versus)\b",
    r"\bhow\s+(?:does|do)\s+.+\s+differ\b",
    r"\bsimilarit(?:y|ies)\s+(?:and|between)\b",
    r"\bcontrast\b",
    r"\bwhich\s+is\s+(?:better|worse|more|less)\b",
    r"\b(.+)\s+or\s+(.+)\s+(?:for|in|with)\b",
]

# Entity extraction patterns for medical comparisons
ENTITY_PATTERNS = [
    # "compare X and Y" or "compare X vs Y"
    r"compare\s+(.+?)\s+(?:and|vs\.?|versus|to|with)\s+(.+?)(?:\s+(?:for|in|regarding)|[.?]|$)",
    # "difference between X and Y"
    r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:for|in|regarding)|[.?]|$)",
    # "X vs Y"
    r"(\b\w+(?:\s+\w+){0,3})\s+(?:vs\.?|versus)\s+(\b\w+(?:\s+\w+){0,3})",
    # "X or Y for/in ..."
    r"(\b\w+(?:\s+\w+){0,2})\s+or\s+(\b\w+(?:\s+\w+){0,2})\s+(?:for|in|with)",
]


@dataclass
class ComparisonEntity:
    """An entity being compared."""
    name: str
    aliases: list[str] = field(default_factory=list)
    context: Optional[str] = None


@dataclass
class ComparisonQuery:
    """A parsed comparison query."""
    original_question: str
    entities: list[ComparisonEntity]
    comparison_type: str  # "versus", "difference", "similarity", "evaluation"
    context_phrase: Optional[str] = None  # e.g., "for pediatric patients"

    def is_valid(self) -> bool:
        """Check if this is a valid comparison with 2+ entities."""
        return len(self.entities) >= 2


@dataclass
class EntityResult:
    """Results for a single entity in a comparison."""
    entity: ComparisonEntity
    chunks: list[dict]
    key_findings: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Complete comparison result with synthesis."""
    query: ComparisonQuery
    entity_results: list[EntityResult]
    synthesis: Optional[str] = None
    confidence_level: str = "medium"
    sources: list[dict] = field(default_factory=list)


def is_comparison_question(question: str) -> bool:
    """
    Detect if a question is asking for a comparison.

    Args:
        question: User's question

    Returns:
        True if this is a comparison question
    """
    question_lower = question.lower()

    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, question_lower):
            return True

    return False


def extract_entities(question: str) -> list[ComparisonEntity]:
    """
    Extract entities being compared from the question.

    Args:
        question: User's question

    Returns:
        List of ComparisonEntity objects
    """
    question_lower = question.lower()
    entities = []

    for pattern in ENTITY_PATTERNS:
        match = re.search(pattern, question_lower, re.IGNORECASE)
        if match:
            # Extract entity names from groups
            for group in match.groups():
                if group and group.strip():
                    entity_name = group.strip().strip(".,?!")
                    # Skip common words
                    if entity_name not in ["the", "a", "an", "it", "this", "that"]:
                        entities.append(ComparisonEntity(
                            name=entity_name,
                            aliases=[entity_name],
                        ))
            break

    # Deduplicate by name
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.name not in seen:
            seen.add(entity.name)
            unique_entities.append(entity)

    return unique_entities[:3]  # Max 3 entities for comparison


def parse_comparison_question(question: str) -> ComparisonQuery:
    """
    Parse a comparison question into structured form.

    Args:
        question: User's question

    Returns:
        ComparisonQuery object
    """
    question_lower = question.lower()

    # Determine comparison type
    if "difference" in question_lower:
        comparison_type = "difference"
    elif "similar" in question_lower:
        comparison_type = "similarity"
    elif any(word in question_lower for word in ["better", "worse", "prefer", "recommend"]):
        comparison_type = "evaluation"
    else:
        comparison_type = "versus"

    # Extract entities
    entities = extract_entities(question)

    # Extract context phrase (e.g., "for pediatric patients")
    context_match = re.search(
        r"(?:for|in|regarding|with)\s+([\w\s]+?)(?:\?|$)",
        question_lower
    )
    context_phrase = context_match.group(1).strip() if context_match else None

    return ComparisonQuery(
        original_question=question,
        entities=entities,
        comparison_type=comparison_type,
        context_phrase=context_phrase,
    )


def generate_subqueries(query: ComparisonQuery) -> list[tuple[ComparisonEntity, str]]:
    """
    Generate sub-queries for each entity in the comparison.

    Args:
        query: Parsed comparison query

    Returns:
        List of (entity, subquery) tuples
    """
    subqueries = []

    base_context = query.context_phrase or "pediatric cancer"

    for entity in query.entities:
        # Generate entity-specific queries
        if query.comparison_type == "difference":
            subquery = f"What are the characteristics of {entity.name} in {base_context}?"
        elif query.comparison_type == "evaluation":
            subquery = f"What are the outcomes and effectiveness of {entity.name} in {base_context}?"
        else:
            subquery = f"{entity.name} treatment outcomes {base_context}"

        subqueries.append((entity, subquery))

    return subqueries


def retrieve_for_comparison(
    query: ComparisonQuery,
    retriever,
    top_k_per_entity: int = 3,
) -> list[EntityResult]:
    """
    Retrieve relevant chunks for each entity in the comparison.

    Args:
        query: Parsed comparison query
        retriever: FAISSRetriever instance
        top_k_per_entity: Number of chunks per entity

    Returns:
        List of EntityResult objects
    """
    results = []
    subqueries = generate_subqueries(query)

    for entity, subquery in subqueries:
        # Search for this entity
        search_results = retriever.search(subquery, top_k=top_k_per_entity)

        # Also try direct entity name search if few results
        if len(search_results) < 2:
            direct_results = retriever.search(
                f"{entity.name} pediatric cancer",
                top_k=top_k_per_entity
            )
            # Merge results, avoiding duplicates
            seen_chunks = {r.chunk_id for r in search_results}
            for r in direct_results:
                if r.chunk_id not in seen_chunks:
                    search_results.append(r)
                    seen_chunks.add(r.chunk_id)

        # Convert to dict format
        chunks = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "chunk_text": r.chunk_text,
                "doc_title": r.doc_title,
                "doc_type": r.doc_type,
                "score": r.score,
                "source_url": r.source_url,
                "section_title": r.section_title,
            }
            for r in search_results
        ]

        results.append(EntityResult(
            entity=entity,
            chunks=chunks,
        ))

    return results


def build_comparison_prompt(
    query: ComparisonQuery,
    entity_results: list[EntityResult],
) -> tuple[str, str]:
    """
    Build the system and user prompts for comparison synthesis.

    Args:
        query: Parsed comparison query
        entity_results: Retrieved results for each entity

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are a research synthesis assistant specializing in pediatric oncology. Your task is to compare findings across different research sources.

Guidelines:
1. Structure your response as a clear comparison
2. For each entity, summarize the key findings from the sources
3. Highlight similarities and differences explicitly
4. Note any contradictions or evolving understanding
5. Cite sources using [Source: Title] format
6. If evidence is limited for one entity, acknowledge this
7. End with a brief synthesis of which option might be preferred (if evaluative) or key takeaways

Response structure:
## [Entity 1 Name]
- Key findings from sources

## [Entity 2 Name]
- Key findings from sources

## Comparison
- Similarities
- Differences
- Key insights

## Confidence
- Assessment of evidence quality"""

    # Build context sections for each entity
    entity_sections = []
    for result in entity_results:
        section_lines = [f"\n### Evidence for: {result.entity.name.upper()}"]

        if not result.chunks:
            section_lines.append("No relevant sources found for this entity.")
        else:
            for i, chunk in enumerate(result.chunks, 1):
                section_lines.append(f"\n[Source {i}: {chunk['doc_title']}]")
                section_lines.append(f"Type: {chunk['doc_type']}")
                section_lines.append(f"Relevance: {chunk['score']:.2f}")
                section_lines.append(chunk['chunk_text'])

        entity_sections.append("\n".join(section_lines))

    context = "\n\n---\n".join(entity_sections)

    user_prompt = f"""## Research Context for Comparison
{context}

---

## Original Question
{query.original_question}

## Comparison Type
{query.comparison_type.title()}

## Instructions
Please provide a structured comparison of {' vs '.join(e.name for e in query.entities)} based on the provided research context. Follow the response structure in your guidelines."""

    return system_prompt, user_prompt


def synthesize_comparison(
    query: ComparisonQuery,
    entity_results: list[EntityResult],
    bedrock_client,
    model_id: str = "amazon.nova-lite-v1:0",
    max_tokens: int = 1500,
) -> ComparisonResult:
    """
    Synthesize a comparison across entities using the LLM.

    Args:
        query: Parsed comparison query
        entity_results: Retrieved results for each entity
        bedrock_client: Boto3 Bedrock runtime client
        model_id: Model ID to use
        max_tokens: Max tokens in response

    Returns:
        ComparisonResult with synthesis
    """
    system_prompt, user_prompt = build_comparison_prompt(query, entity_results)

    # Build request based on model type
    if "anthropic" in model_id or "claude" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        })
    else:
        # Amazon Nova format
        body = json.dumps({
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        })

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())

    # Parse response based on model type
    if "anthropic" in model_id or "claude" in model_id:
        synthesis = response_body["content"][0]["text"]
    else:
        synthesis = response_body["output"]["message"]["content"][0]["text"]

    # Collect all sources
    all_sources = []
    seen_docs = set()
    for result in entity_results:
        for chunk in result.chunks:
            if chunk["document_id"] not in seen_docs:
                seen_docs.add(chunk["document_id"])
                all_sources.append({
                    "document_id": chunk["document_id"],
                    "doc_title": chunk["doc_title"],
                    "doc_type": chunk["doc_type"],
                    "score": chunk["score"],
                    "source_url": chunk.get("source_url"),
                    "entity": result.entity.name,
                })

    # Compute confidence based on coverage
    total_chunks = sum(len(r.chunks) for r in entity_results)
    entities_with_results = sum(1 for r in entity_results if r.chunks)

    if entities_with_results == len(entity_results) and total_chunks >= 4:
        confidence_level = "high"
    elif entities_with_results >= 1 and total_chunks >= 2:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return ComparisonResult(
        query=query,
        entity_results=entity_results,
        synthesis=synthesis,
        confidence_level=confidence_level,
        sources=all_sources,
    )


def handle_comparison_question(
    question: str,
    retriever,
    bedrock_client,
    model_id: str = "amazon.nova-lite-v1:0",
    top_k_per_entity: int = 3,
) -> dict:
    """
    Handle a comparison question end-to-end.

    Args:
        question: User's comparison question
        retriever: FAISSRetriever instance
        bedrock_client: Boto3 Bedrock client
        model_id: LLM model ID
        top_k_per_entity: Chunks per entity

    Returns:
        Response dict with answer, sources, and metadata
    """
    # Parse the comparison question
    query = parse_comparison_question(question)

    if not query.is_valid():
        logger.warning(f"Could not extract comparison entities from: {question}")
        return {
            "is_comparison": False,
            "message": "Could not identify entities to compare. Please rephrase with explicit entities (e.g., 'Compare X and Y').",
        }

    logger.info(f"Comparison query: {[e.name for e in query.entities]}, type={query.comparison_type}")

    # Retrieve for each entity
    entity_results = retrieve_for_comparison(query, retriever, top_k_per_entity)

    # Synthesize comparison
    result = synthesize_comparison(
        query=query,
        entity_results=entity_results,
        bedrock_client=bedrock_client,
        model_id=model_id,
    )

    return {
        "is_comparison": True,
        "question": question,
        "entities": [e.name for e in query.entities],
        "comparison_type": query.comparison_type,
        "answer": result.synthesis,
        "confidence": result.confidence_level,
        "sources": result.sources,
        "entity_coverage": {
            r.entity.name: len(r.chunks) for r in entity_results
        },
    }
