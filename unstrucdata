from datetime import date
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field


# ============================================================
# 1. PYDANTIC MODEL
# ============================================================

class DocumentMetadata(BaseModel):

    title: str = Field(
        ...,
        description="Title of the document"
    )

    author: str = Field(
        ...,
        description="Author of the document"
    )

    publication_date: date = Field(
        ...,
        description="Publication date in YYYY-MM-DD format"
    )

    keywords: list[str] = Field(
        ...,
        description="List of keywords"
    )

    document_type: str = Field(
        ...,
        description="Type of document, such as report or article"
    )


# ============================================================
# 2. JSON OUTPUT PARSER
# ============================================================

parser = JsonOutputParser(
    pydantic_object=DocumentMetadata
)


# ============================================================
# 3. PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a metadata extraction system.

Read the document below and extract its metadata.

You MUST return ONLY a JSON object.

Do NOT return:
- explanations
- markdown
- ```json
- the JSON schema
- properties
- required
- descriptions

Return exactly these five fields:

{{
    "title": "document title",
    "author": "document author",
    "publication_date": "YYYY-MM-DD",
    "keywords": ["keyword1", "keyword2"],
    "document_type": "report or article"
}}

Document:

{document_text}
""",

    input_variables=["document_text"]
)


# ============================================================
# 4. OLLAMA MODEL
# ============================================================

model = ChatOllama(
    model="gemma3:1b",
    temperature=0
)


# ============================================================
# 5. DOCUMENTS
# ============================================================

documents = [

    """
    This report, 'AI Trends 2025', was written by Dr. Sarah Lee
    and published on 2025-05-01.

    It covers topics like artificial intelligence,
    deep learning, and ethics in AI.
    """,

    """
    The article 'Climate Change and Agriculture' by John Smith
    was released on 2024-09-15.

    It discusses sustainability, farming practices,
    and environmental policy.
    """
]


# ============================================================
# 6. STRUCTURED DATASET
# ============================================================

structured_metadata = []


# ============================================================
# 7. PROCESS DOCUMENTS
# ============================================================

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 60)
    print(f"PROCESSING DOCUMENT {index}")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Create prompt
        # ----------------------------------------------------

        prompt_text = prompt.format(
            document_text=document
        )

        # ----------------------------------------------------
        # Call Ollama
        # ----------------------------------------------------

        response = model.invoke(
            prompt_text
        )

        print("\nRAW LLM RESPONSE:")
        print(response.content)

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        parsed_data = parser.parse(
            response.content
        )

        print("\nPARSED DATA:")
        print(parsed_data)

        # ----------------------------------------------------
        # Validate with Pydantic
        # ----------------------------------------------------

        metadata = DocumentMetadata.model_validate(
            parsed_data
        )

        # ----------------------------------------------------
        # Convert Pydantic object to dictionary
        # ----------------------------------------------------

        data = metadata.model_dump(
            mode="json"
        )

        # ----------------------------------------------------
        # Add to dataset
        # ----------------------------------------------------

        structured_metadata.append(
            data
        )

        print("\nVALIDATION: SUCCESS")

    except Exception as error:

        print("\nVALIDATION: FAILED")
        print(error)


# ============================================================
# 8. FINAL DATASET
# ============================================================

print("\n\n")
print("=" * 60)
print("FINAL STRUCTURED DATASET")
print("=" * 60)

print(
    json.dumps(
        structured_metadata,
        indent=4,
        ensure_ascii=False
    )
)