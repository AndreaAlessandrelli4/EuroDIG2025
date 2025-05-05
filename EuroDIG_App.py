import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from huggingface_hub import login
#from transformers import pipeline
from utils import (
    extract_text_from_pdf,
    cleaning_page,
    remove_unicode_characters,
    cleaning,
    connection,
    quering_database,
)



# --- Page config & CSS ---
st.set_page_config(page_title="Justice Indexer", layout="wide")
st.markdown(
    """
    <style>
        /* Global */
        body, .stApp { background-color: #f5f7fa !important; }
        h1, h2, h3, h4, h5, h6 { color: #003399 !important; }
        p, div { color: #333 !important; }
        hr { border: 1px solid #003399 !important; }

        /* Sidebar */
        .stSidebar { background-color: #003399 !important; color: white !important; }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar div { color: white !important; }

        /* Buttons */
        .stButton>button, .stDownloadButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            padding: 1rem 2rem;
            border-radius: 8px;
            font-size: 1.1rem;
        }
        .stButton>button:hover { background-color: #e6c200 !important; }

        /* Expander */
        div.stExpander {
            background-color: #0056b3 !important;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #FFD700;
        }
        div.stExpander * { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize page state ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

with st.sidebar:
    # --- Top-right navigation selectbox ---
    col1, col2 = st.columns([1,0.5])
    with col1:
        page_select = st.selectbox(
            "", ["Overview", "Tool"],
            index=["Overview", "Tool"].index(st.session_state.current_page),
            key="nav_select"
        )
        if page_select != st.session_state.current_page:
            st.session_state.current_page = page_select
    page = st.session_state.current_page

# --- Helper: load models ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
    rer = FlagReranker("BAAI/bge-reranker-large", device=device)
    return emb, rer

# --- Overview page ---
if page == 'Overview':
    # Header images
    c1, c2, c3, c4, c5 = st.columns([0.5, 0.5, 1, 0.5, 0.5])
    with c1:
        st.image("/Foto/COE-Logo-Quadri.png", use_container_width=True)
    with c5:
        st.image("/Foto/euroAVVOVChat.png", use_container_width=True)
    with c3:
        # Title & subtitle
        st.markdown("<h1 style='text-align:center;'>LexiFind</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;font-size:1.2rem;color:#555;'>Your Advanced Legal Search Engine</p>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border:1px solid #007bff;'>", unsafe_allow_html=True)

    # Features
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("### 🔍 Flexible Query Input")
        st.markdown(
            "- **Text Query:** Type your keywords or case summary.\n"
            "- **Document Upload:** Upload a judgment, memorandum, or brief."
        )
    with colB:
        st.markdown("### ⚙️ Customizable Results")
        st.markdown(
            "- **Max Results:** Select Top 10, 25, 50, etc.\n"
            "- **Search Type:** Keyword (exact) or Hybrid (semantic)."
        )
    with colC:
        st.markdown("### 📄 Detailed Output")
        st.markdown(
            "- **Metadata:** Date, jurisdiction, document type.\n"
            "- **Summary:** Parties, facts, ruling.\n"
            "- **Download:** One-click full-text access."
        )

    st.markdown("<hr style='border:1px solid #007bff;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>🔄 Keyword vs. Hybrid Search</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <table style='width:100%;border-collapse:collapse;'>
          <tr style='background:#f5f5f5;'><th>Mode</th><th>Use Case</th></tr>
          <tr><td><strong>Keyword</strong></td><td>Exact phrases, specific legal terms.</td></tr>
          <tr><td><strong>Hybrid</strong></td><td>Conceptual searches across formats/languages.</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    
    # CTA button navigates to Search
    st.markdown("<div style='text-align:center; margin:2rem 0;'>", unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    if middle.button("Start Your Search Now ", icon="🔎", use_container_width=True):
        st.session_state.current_page = 'Tool'
    st.markdown("</div>", unsafe_allow_html=True)

    # The Team section
    st.markdown("<hr style='border:1px solid #007bff;'>", unsafe_allow_html=True)
    team_cols1 = st.columns([1,1,0.7,0.3,0.7,0.3,0.7,1,1,0.1])
    with team_cols1[4]:
        st.image("/Foto/logo.png", use_container_width=True)
    st.markdown("<h3 style='text-align:center;'>The Team</h3>", unsafe_allow_html=True)
    
    team_cols = st.columns([1,1,0.7,0.3,0.7,0.3,0.7,1,1,0.1])
    with team_cols[2]:
        st.image("/Foto/Pasquale.png", use_container_width=True)
        st.markdown(
            """
            <div style='text-align:center;'>
              <a href='https://linkedin.com/in/pasquale-maritato-111074218' target='_blank'><strong>Pasquale Maritato</strong></a>
              <p style='margin:0;'>Fraud Data Scientist, Poste Italiane</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with team_cols[4]:
        st.image("/Foto/Andrea.png", use_container_width=True)
        st.markdown(
            """
            <div style='text-align:center;'>
              <a href='https://www.linkedin.com/in/andrea-alessandrelli4/' target='_blank'><strong>Andrea Alessandrelli</strong></a>
              <p style='margin:0;'>PhD student in AI, University of Pisa</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with team_cols[6]:
        st.image("/Foto/Fabrizio.png", use_container_width=True)
        st.markdown(
            """
            <div style='text-align:center;'>
              <a href='https://www.linkedin.com/in/fabrizio-tomasso/' target='_blank'><strong>Fabrizio Tomasso</strong></a>
              <p style='margin:0;'>Project Manager, Molise</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Search page ---
else:
    # Load models
    # --- Hugging Face login ---
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    login(hf_token)
    emb_model, rer_model = load_models()
    with st.sidebar:
        st.markdown("Our solution is a robust system designed to retrieve judicial court cases related to user-inputted prompts." 
                    "\n\nIt utilizes an open-source vector database, Weaviate, which supports hybrid searches that combine embedding representations of court cases with keyword searches to deliver relevant documents."
                    "\n\nTo handle the multilingual nature of legal datasets, we employ a multilingual embedding model. For keyword and summary extraction, we use the **meta-llama/Llama-3.1-8B** model due to its speed and ability to efficiently handle long texts in multiple languages. To rerank the results, we use a multilingual reranker")
        st.markdown("### ⚙️ Key Components of the Project:")
        st.markdown(
            "- **Vector Representation Storage:** Store vector representations of court cases in Weaviate.\n"
            "- **⁠Keyword Representation Storage:** Store keyword representations of court cases in English within Weaviate.\n"
            "- **Hybrid Search::** Retrieve court cases with vector and keyword searches.\n"
            "- **⁠Reranking Mechanism:** Develop a reranker to prioritize the retrieved court cases effectively.\n"
        )

    c1, c2, c3, c4, c5 = st.columns([0.5, 0.5, 1, 0.5, 0.5])
    with c1:
        st.image("/Foto/euroAVVOVChat.png", use_container_width=True)
    with c5:
        st.image("/Foto/logo.png", use_container_width=True)
    with c3:
        # Title & subtitle
        st.markdown("<h1 style='text-align:center;'>LexiFind</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;font-size:1.2rem;color:#555;'>Your Advanced Legal Search Engine</p>",
            unsafe_allow_html=True,
        )
    # Sidebar settings for search
    st.markdown("<h6 style='text-align:left;font-size:1.1rem;color:#003399;'>📄 Input method:</h6>",unsafe_allow_html=True,)
    input_mode = st.radio(":", ["Query", "Upload document (PDF/TXT)"], horizontal=True , label_visibility='collapsed')

    # Query input area
    query = ""
    if input_mode == "Query":
        query = st.text_area("Enter your query:", label_visibility='collapsed', placeholder="Enter your query", height=100)
        #query = st.text_input("Enter your query:", label_visibility='collapsed', placeholder="Enter your query")
    else:
        uploaded = st.file_uploader("Upload a document (PDF/TXT)", type=["pdf", "txt"], label_visibility='collapsed')
        if uploaded:
            raw = extract_text_from_pdf(uploaded) if uploaded.type == "application/pdf" else uploaded.read().decode("utf-8")
            cleaned = remove_unicode_characters(cleaning(cleaning_page(raw)))
            if "Errore" in cleaned:
                st.error(cleaned)
            else:
                query = cleaned
                st.text_area("Extracted text:", query, height=200)

    # Execute search
    cc=st.columns([3,1,1,2])
    with cc[0]:
        st.markdown("<h6 style='text-align:left;font-size:1.1rem;color:#003399;'>🔎 Select the type of the Search</h6>",unsafe_allow_html=True,)
        type_search = st.radio(":", ["Keyword Search", "Hybrid Search"], horizontal=True , label_visibility='collapsed')
    with cc[-1]:
        st.markdown("<h6 style='text-align:left;font-size:1.1rem;color:#003399;'>Max Number of results:</h6>",unsafe_allow_html=True,)
        num_results = st.number_input("Max Number of results:", 1, 20, 10, label_visibility='collapsed')
    search_type = "key" if type_search == "Keyword Search" else "Hybrid Search"

    
    if st.button(type_search):
        if query =="":
            col = st.columns([0.5,1,0.5])
            with col[1]:
                st.markdown("<h4 style='text-align:center; background-color: #F6F3D8;  padding: 5px; border: 2px solid #FFD700;'>🚨 No query or documents inserted 🚨</h4>", unsafe_allow_html=True)
        else:
            URL = os.getenv("WEAVIATE_URL_summary")
            KEY = os.getenv("WEAVIATE_API_KEY_summary")
            client = connection(URL, KEY)
            _, _, results = quering_database(
                query, rer_model, emb_model, client, "Euro2025", search_type, max_doc=num_results
            )
            if results:
                for r in results:
                    date = str(r["date"])[:10]
                    st.markdown(f"## {r['court']} — {date}")
                    with st.expander("💬 Metadata"):
                        st.write(f"**Language:** {r['language']}")
                        st.write(f"**Data:** {str(r['date'])[0:10]}")
                        st.write(f"**Type:** {r['type']}")
                        st.write(f"**Court:** {r['court']}")
                        st.write(f"**Keywords:** {r['description']}")
                    with st.expander("📜 Summary"):
                        st.write(r['summary'].strip().lstrip('#').replace('**Summary**','').replace('Summary',''))
                    with st.expander("📑 Full Text"):
                        st.download_button(
                            "📥 Download full text",
                            r['full_text'],
                            file_name=f"Case_{r['court']}_{date}.txt",
                            mime="text/plain",
                        )
                        st.write(r['full_text'])
                    st.markdown("---")
            else:
                col = st.columns([0.5,1,0.5])
                with col[1]:
                    st.markdown("<h5 style='text-align:center; background-color: #F6F3D8;  padding: 6px; border: 2px solid #FFD700;'>🚨 No results compatible with the search criteria 🚨</h5>", unsafe_allow_html=True)
    # Implicit back via top-right navigation
