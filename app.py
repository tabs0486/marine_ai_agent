import os
import re

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

from prompts import build_rag_prompt
from workflow import run_rag_workflow


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marine AI Troubleshooting Agent",
    page_icon="⚓",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("⚓ Marine AI Troubleshooting Agent")

st.markdown(
    """
AI-powered marine engine troubleshooting assistant using:

**RAG + FAISS + Sentence Transformers + PyMuPDF + Groq**

Upload an approved technical manual, enter the engine information,
describe the defect/alarm, and ask your troubleshooting question.
"""
)

st.warning(
    """
SAFETY NOTICE

This application provides AI-assisted information retrieval and
troubleshooting guidance. Always verify recommendations against
the exact OEM manual, vessel SMS, class requirements and qualified
engineering personnel.

Never bypass alarms, trips, interlocks or safety systems based
solely on an AI response.
"""
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def create_groq_client(api_key):

    return Groq(
        api_key=api_key
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        if text and text.strip():

            cleaned_text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            pages.append(
                {
                    "page": page_number,
                    "text": cleaned_text
                }
            )

    document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=900,
    overlap=150
):

    chunks = []

    for page in pages:

        words = page["text"].split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )

            if end >= len(words):
                break

            start = max(
                end - overlap,
                start + 1
            )

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    chunks,
    embedding_model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    query,
    index,
    chunks,
    embedding_model,
    top_k=6
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        results.append(
            {
                "text": chunks[index_number]["text"],
                "page": chunks[index_number]["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(
    retrieved_chunks
):

    context = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context.append(
            f"""
[SOURCE {number}
PDF PAGE: {chunk['page']}
SIMILARITY: {chunk['score']:.3f}]

{chunk['text']}
"""
        )

    return "\n\n".join(context)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        value=st.secrets.get(
            "GROQ_API_KEY",
            os.getenv(
                "GROQ_API_KEY",
                ""
            )
        ),
        type="password"
    )

    top_k = st.slider(
        "Number of manual sections to retrieve",
        min_value=3,
        max_value=10,
        value=6
    )

    st.markdown("---")

    st.markdown(
        """
### RAG Pipeline

PDF
↓
PyMuPDF
↓
Chunks
↓
Sentence Transformer
↓
Embeddings
↓
FAISS
↓
Relevant Manual Sections
↓
Groq
↓
Troubleshooting Answer
"""
    )


# ============================================================
# ENGINE INFORMATION
# ============================================================

st.header("1️⃣ Engine Information")

col1, col2, col3 = st.columns(3)

with col1:

    manufacturer = st.text_input(
        "Engine Manufacturer",
        placeholder="Example: MAN"
    )

with col2:

    engine_model = st.text_input(
        "Engine Model",
        placeholder="Example: 6L48/60"
    )

with col3:

    serial_number = st.text_input(
        "Engine Serial Number",
        placeholder="Optional but recommended"
    )


# ============================================================
# DEFECT
# ============================================================

st.header("2️⃣ Engine Defect / Alarm")

defect = st.text_area(
    "Describe the engine problem or alarm",
    placeholder="""
Example:

Main engine has low lube oil pressure at normal operating RPM.
Alarm LO-LOW appears after the engine reaches operating temperature.
""",
    height=150
)


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("3️⃣ Technical Manual")

uploaded_pdf = st.file_uploader(
    "Upload the relevant OEM / technical manual",
    type=["pdf"]
)


if uploaded_pdf:

    st.info(
        f"Selected manual: {uploaded_pdf.name}"
    )

    if st.button(
        "🔎 Build RAG Knowledge Base"
    ):

        try:

            # -------------------------
            # Extract PDF
            # -------------------------

            with st.spinner(
                "Extracting PDF text..."
            ):

                pages = extract_pdf_text(
                    uploaded_pdf
                )

            if not pages:

                st.error(
                    """
No selectable text was found in this PDF.

The PDF may be scanned/image-based.

For this version, please use a text-based PDF.
"""
                )

                st.stop()

            # -------------------------
            # Create chunks
            # -------------------------

            with st.spinner(
                "Creating document chunks..."
            ):

                chunks = create_chunks(
                    pages
                )

            # -------------------------
            # Embeddings
            # -------------------------

            with st.spinner(
                "Creating embeddings..."
            ):

                embedding_model = (
                    load_embedding_model()
                )

                index = create_faiss_index(
                    chunks,
                    embedding_model
                )

            # -------------------------
            # Save in session
            # -------------------------

            st.session_state[
                "pages"
            ] = pages

            st.session_state[
                "chunks"
            ] = chunks

            st.session_state[
                "faiss_index"
            ] = index

            st.session_state[
                "manual_name"
            ] = uploaded_pdf.name

            st.success(
                f"""
RAG knowledge base created successfully.

Pages: {len(pages)}

Chunks: {len(chunks)}
"""
            )

        except Exception as error:

            st.error(
                f"PDF processing error: {error}"
            )


# ============================================================
# QUESTION
# ============================================================

st.header("4️⃣ Ask the Marine AI Agent")

question = st.text_area(
    "Your troubleshooting question",
    placeholder="""
Example:

What are the most likely causes of this low lube oil pressure?
What should I check first?
What does the manual recommend?
""",
    height=130
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "⚓ Analyze Engine Defect",
    type="primary",
    use_container_width=True
):

    # -------------------------
    # Validation
    # -------------------------

    if not api_key:

        st.error(
            "Groq API key is missing."
        )

        st.stop()

    if not manufacturer:

        st.error(
            "Please enter the engine manufacturer."
        )

        st.stop()

    if not engine_model:

        st.error(
            "Please enter the engine model."
        )

        st.stop()

    if not defect and not question:

        st.error(
            "Please enter an engine defect or question."
        )

        st.stop()

    if (
        "faiss_index"
        not in st.session_state
    ):

        st.error(
            """
Please upload the technical manual
and click "Build RAG Knowledge Base"
first.
"""
        )

        st.stop()

    # -------------------------
    # Load embedding model
    # -------------------------

    embedding_model = (
        load_embedding_model()
    )

    # -------------------------
    # Build search query
    # -------------------------

    search_query = f"""
Engine Manufacturer:
{manufacturer}

Engine Model:
{engine_model}

Engine Serial Number:
{serial_number}

Engine Defect:
{defect}

Question:
{question}
"""

    # -------------------------
    # Retrieve documents
    # -------------------------

    with st.spinner(
        "Searching the technical manual..."
    ):

        retrieved_chunks = (
            retrieve_relevant_chunks(
                search_query,
                st.session_state[
                    "faiss_index"
                ],
                st.session_state[
                    "chunks"
                ],
                embedding_model,
                top_k
            )
        )

    # -------------------------
    # Build context
    # -------------------------

    context = build_context(
        retrieved_chunks
    )

    # -------------------------
    # Groq
    # -------------------------

    with st.spinner(
        "AI is analyzing the defect..."
    ):

        client = create_groq_client(
            api_key
        )

        result = run_rag_workflow(
            client=client,
            manufacturer=manufacturer,
            engine_model=engine_model,
            serial_number=serial_number,
            defect=defect,
            question=question,
            context=context
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    st.header(
        "🧭 Troubleshooting Assessment"
    )

    st.markdown(
        result
    )

    # ========================================================
    # SOURCES
    # ========================================================

    st.header(
        "📚 Retrieved Manual Evidence"
    )

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        with st.expander(
            f"""
Source {number}
|
PDF Page {chunk['page']}
|
Similarity {chunk['score']:.3f}
"""
        ):

            st.write(
                chunk["text"]
            )

    st.caption(
        f"""
Manual: {st.session_state.get('manual_name', 'Unknown')}
|
Retrieved sections: {len(retrieved_chunks)}
"""
    )


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "ℹ️ How the system works"
):

    st.markdown(
        """
### 1. PDF extraction

PyMuPDF extracts text from the uploaded technical manual.

### 2. Chunking

The manual is divided into smaller overlapping sections.

### 3. Embeddings

Sentence Transformer converts each section into a numerical vector.

### 4. FAISS

FAISS stores the vectors and performs similarity search.

### 5. Retrieval

The user's defect and question are converted into an embedding.

The system retrieves the most relevant manual sections.

### 6. Generation

The retrieved evidence is sent to Groq's
`openai/gpt-oss-120b` model.

### 7. Grounded answer

The model produces a structured troubleshooting response
based on the retrieved technical documentation.
"""
    )
