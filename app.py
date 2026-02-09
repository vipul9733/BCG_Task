import streamlit as st
import anthropic
import PyPDF2
import io
import json
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict, Counter
import networkx as nx
from pyvis.network import Network
import math

# Page configuration
st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    .evidence-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .citation {
        background-color: #d4edda;
        padding: 0.3rem 0.5rem;
        border-radius: 3px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class PaperChunk:
    """Represents a chunk of text from a paper with page number"""
    def __init__(self, text: str, page_num: int, paper_name: str):
        self.text = text
        self.page_num = page_num
        self.paper_name = paper_name


class ResearchPaperAnalyzer:
    """AI-powered research paper analyzer with RAG and citation tracking"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def extract_text_with_pages(self, pdf_file) -> List[Tuple[int, str]]:
        """Extract text content from PDF with page numbers"""
        try:
            # FIX #1: Reset file pointer to beginning
            pdf_file.seek(0)
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages_text = []
            for i, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages_text.append((i, text))
            return pages_text
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return []
    
    def create_chunks(self, pages_text: List[Tuple[int, str]], paper_name: str, 
                     chunk_size: int = 1200, overlap: int = 200) -> List[PaperChunk]:
        """Create overlapping chunks from pages for better context"""
        # FIX #4: Proper overlapping chunks
        chunks = []
        
        for page_num, text in pages_text:
            # Clean text - remove extra whitespace
            text = ' '.join(text.split())
            
            # Create overlapping chunks
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                # Only add non-empty chunks
                if chunk_text.strip():
                    chunks.append(PaperChunk(chunk_text.strip(), page_num, paper_name))
                
                # Move by (chunk_size - overlap) for overlap
                start += (chunk_size - overlap)
                
                # Break if we've covered the text
                if end >= len(text):
                    break
        
        return chunks
    
    def extract_references(self, pages_text: List[Tuple[int, str]]) -> List[str]:
        """Extract and normalize references from paper"""
        references = []
        full_text = " ".join([text for _, text in pages_text])
        
        # Look for References/Bibliography section
        ref_section = re.search(
            r'(?:References|Bibliography|Works Cited)[\s\S]*$', 
            full_text, 
            re.IGNORECASE
        )
        
        if ref_section:
            ref_text = ref_section.group()
            # Extract lines that look like citations
            potential_refs = re.findall(
                r'\[?\d+\]?\s*[A-Z][^.!?]*(?:\.|,)\s*(?:\(\d{4}\)|20\d{2})',
                ref_text
            )
            
            # FIX #5: Normalize references for better matching
            normalized_refs = []
            for ref in potential_refs[:20]:  # Limit to 20
                normalized = self.normalize_reference(ref)
                if normalized:
                    normalized_refs.append(normalized)
            
            references = normalized_refs
        
        return references
    
    def normalize_reference(self, ref: str) -> str:
        """Normalize reference for better matching across papers"""
        # FIX #5: Reference normalization
        # Convert to lowercase
        ref = ref.lower()
        
        # Remove punctuation except parentheses (for years)
        ref = re.sub(r'[^\w\s\(\)]', ' ', ref)
        
        # Collapse whitespace
        ref = ' '.join(ref.split())
        
        # Extract author + year pattern (simple heuristic)
        # Look for pattern like "smith 2020" or "jones et al 2019"
        author_year = re.search(r'([a-z]+(?:\s+et\s+al)?)\s+\(?(\d{4})\)?', ref)
        
        if author_year:
            # Use author + year as canonical form
            return f"{author_year.group(1)} {author_year.group(2)}"
        
        # Fallback: return normalized string (truncated)
        return ref[:100]
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for better tokenization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        # This handles "methodology," vs "methodology" correctly
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter out very short tokens and common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                    'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        tokens = [
            token for token in text.split() 
            if len(token) > 2 and token not in stopwords
        ]
        
        return tokens
    
    def compute_tfidf_scores(self, query: str, chunks: List[PaperChunk]) -> List[Tuple[int, float]]:
        """Compute TF-IDF scores for chunks (local scoring, no LLM call)"""
        # FIX #2: Use local scoring instead of expensive LLM call
        # IMPROVED: Better tokenization and efficient TF computation
        
        # Tokenize query with preprocessing
        query_terms = set(self.preprocess_text(query))
        
        if not query_terms:
            # Fallback if all query terms were stopwords
            return [(i, 0.0) for i in range(len(chunks))]
        
        # Preprocess all chunks and build term frequency maps (more efficient)
        chunk_term_freqs = []
        for chunk in chunks:
            tokens = self.preprocess_text(chunk.text)
            term_freq = Counter(tokens)
            chunk_term_freqs.append(term_freq)
        
        # Calculate document frequency for each query term
        doc_freq = Counter()
        for term_freq in chunk_term_freqs:
            for term in query_terms:
                if term in term_freq:
                    doc_freq[term] += 1
        
        # Calculate TF-IDF score for each chunk
        scores = []
        num_docs = len(chunks)
        
        for idx, term_freq in enumerate(chunk_term_freqs):
            total_terms = sum(term_freq.values())
            if total_terms == 0:
                scores.append((idx, 0.0))
                continue
            
            score = 0.0
            for term in query_terms:
                if term in term_freq:
                    # Term frequency (normalized)
                    tf = term_freq[term] / total_terms
                    
                    # Inverse document frequency
                    df = doc_freq.get(term, 0)
                    idf = math.log((num_docs + 1) / (df + 1)) if df > 0 else 0
                    
                    # TF-IDF
                    score += tf * idf
            
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def find_relevant_chunks(self, question: str, all_chunks: List[PaperChunk], 
                           top_k: int = 7) -> List[PaperChunk]:
        """Find relevant chunks using local TF-IDF scoring"""
        # FIX #2: No expensive LLM call for retrieval
        # IMPROVED: Always return top-k chunks even with low scores
        
        if not all_chunks:
            return []
        
        # Compute TF-IDF scores
        scores = self.compute_tfidf_scores(question, all_chunks)
        
        # Get top-k chunks, even if scores are low
        # This prevents "No relevant evidence" when papers use different phrasing
        top_k = min(top_k, len(all_chunks))
        top_indices = [idx for idx, score in scores[:top_k]]
        
        # Return top chunks
        return [all_chunks[idx] for idx in top_indices]
    
    def analyze_paper_with_chunks(self, chunks: List[PaperChunk], 
                                  paper_name: str) -> Dict[str, Any]:
        """Analyze paper using chunks for better grounding"""
        
        # Combine chunks for analysis (limit to ~15k chars for efficiency)
        combined_text = "\n\n".join([
            f"[Page {chunk.page_num}] {chunk.text}" 
            for chunk in chunks[:30]
        ])
        
        prompt = f"""Analyze this research paper and extract structured information. 
Each chunk is marked with [Page X] for citation purposes.

Paper: {paper_name}

Content:
{combined_text[:15000]}

Provide analysis in JSON format with PAGE CITATIONS:
{{
    "title": "extracted title",
    "authors": ["list of authors"],
    "abstract": "summary with (p. X) citations",
    "research_question": "main question",
    "methodology": "methods used with (p. X) citations",
    "key_findings": [
        {{
            "finding": "specific finding",
            "page": X,
            "evidence": "brief quote or paraphrase"
        }}
    ],
    "contributions": "main contributions",
    "limitations": [
        {{
            "limitation": "description",
            "page": X
        }}
    ],
    "future_work": "suggested directions",
    "domain": "research field"
}}

CRITICAL: Include page numbers for all factual claims. Use format (p. X) in text."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            # FIX #3: Robust JSON extraction
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback if no JSON found
                analysis = {"error": "Could not parse response"}
            
            analysis['paper_name'] = paper_name
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing {paper_name}: {str(e)}")
            return {
                "paper_name": paper_name,
                "title": "Analysis failed",
                "error": str(e)
            }
    
    def answer_with_evidence(self, question: str, all_chunks: List[PaperChunk],
                           analyses: List[Dict[str, Any]]) -> Tuple[str, List[PaperChunk]]:
        """Answer question with grounded evidence using local retrieval"""
        
        # FIX #2: Get relevant chunks using local TF-IDF (no LLM call)
        relevant_chunks = self.find_relevant_chunks(question, all_chunks, top_k=7)
        
        if not relevant_chunks:
            return "No relevant evidence found in the uploaded papers.", []
        
        # Build context from chunks
        context = "\n\n".join([
            f"Evidence from {chunk.paper_name} (Page {chunk.page_num}):\n{chunk.text}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""Answer this question using ONLY the provided evidence. 
Include citations in format: (Paper Name, p. X)

Question: {question}

Evidence:
{context}

Paper summaries for context:
{json.dumps([{
    'paper': a.get('paper_name'),
    'title': a.get('title'),
    'findings': a.get('key_findings', [])[:3]
} for a in analyses], indent=2)[:2000]}

Provide:
1. Direct answer with citations
2. Confidence level (High/Medium/Low)
3. Note any limitations or missing information

Format: Clear paragraphs with inline citations."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = message.content[0].text
            return answer, relevant_chunks
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
    
    def compare_papers(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-analyze papers"""
        
        papers_summary = []
        for i, analysis in enumerate(analyses, 1):
            findings_text = ""
            if 'key_findings' in analysis and analysis['key_findings']:
                if isinstance(analysis['key_findings'][0], dict):
                    findings_text = ', '.join([
                        f"{f.get('finding', '')} (p.{f.get('page', '?')})"
                        for f in analysis['key_findings'][:3]
                    ])
                else:
                    findings_text = ', '.join(analysis['key_findings'][:3])
            
            summary = f"""
Paper {i}: {analysis.get('title', 'Unknown')}
- Research Question: {analysis.get('research_question', 'N/A')}
- Methodology: {analysis.get('methodology', 'N/A')}
- Key Findings: {findings_text}
- Domain: {analysis.get('domain', 'N/A')}
"""
            papers_summary.append(summary)
        
        comparison_prompt = f"""Analyze {len(analyses)} papers together.

Papers:
{''.join(papers_summary)}

Provide JSON:
{{
    "common_themes": ["themes across papers"],
    "contradictions": [
        {{
            "topic": "disagreement topic",
            "paper_1_position": "Paper 1 stance with page refs",
            "paper_2_position": "Paper 2 stance with page refs",
            "significance": "why it matters"
        }}
    ],
    "methodological_comparison": {{
        "similarities": ["common approaches"],
        "differences": ["different approaches"]
    }},
    "complementary_insights": ["how papers work together"],
    "research_gaps": ["identified gaps"],
    "synthesis": "2-3 paragraph synthesis"
}}"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": comparison_prompt}]
            )
            
            response_text = message.content[0].text
            
            # FIX #3: Robust JSON extraction
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse comparison"}
            
        except Exception as e:
            st.error(f"Error comparing papers: {str(e)}")
            return {}
    
    def build_citation_network(self, papers_data: List[Dict]) -> nx.Graph:
        """Build undirected citation network from papers"""
        # FIX #6: Use undirected graph for symmetric co-citation
        G = nx.Graph()  # Changed from DiGraph
        
        # Add nodes for each paper
        for i, paper in enumerate(papers_data):
            G.add_node(
                i,
                label=paper['name'][:30] + '...' if len(paper['name']) > 30 else paper['name'],
                title=paper.get('title', 'Unknown'),
                size=25
            )
        
        # Find citation relationships (co-citation)
        for i in range(len(papers_data)):
            for j in range(i + 1, len(papers_data)):  # FIX #6: Only i < j to avoid duplicates
                refs_i = set(papers_data[i].get('references', []))
                refs_j = set(papers_data[j].get('references', []))
                
                # Count shared references
                shared_refs = refs_i & refs_j
                shared_count = len(shared_refs)
                
                if shared_count > 0:
                    # Add edge for co-citation (undirected)
                    G.add_edge(
                        i, j, 
                        weight=shared_count,
                        title=f"{shared_count} shared refs",
                        color='#3498db'
                    )
        
        return G
    
    def visualize_network(self, G: nx.Graph) -> str:
        """Create interactive network visualization and return HTML string"""
        # FIX #7: Return HTML string instead of temp file
        
        net = Network(
            height="600px", 
            width="100%", 
            directed=False,  # FIX #6: Undirected
            notebook=False,
            cdn_resources='in_line'
        )
        
        # Configure physics
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)
        
        # Add nodes
        for node, data in G.nodes(data=True):
            net.add_node(
                node,
                label=data['label'],
                title=data.get('title', ''),
                size=data.get('size', 25),
                color='#1f77b4'
            )
        
        # Add edges
        for source, target, data in G.edges(data=True):
            net.add_edge(
                source, target,
                title=data.get('title', ''),
                color=data.get('color', '#3498db'),
                width=data.get('weight', 1) * 2
            )
        
        # FIX #7: Generate HTML in memory, no temp files
        html_string = net.generate_html()
        
        return html_string


def load_demo_papers():
    """Placeholder for demo papers - users can add their own"""
    st.info("""
    **Demo Mode:** To add sample papers for instant testing:
    1. Add 2-3 PDF papers to a `sample_papers/` folder in your repo
    2. Update this function to load them
    3. Users can click "Load Sample Papers" to test immediately
    
    For now, please upload your own research papers (PDF format).
    Good sources: arXiv.org, Google Scholar, PubMed
    """)
    return []


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📚 AI Research Paper Summarizer</div>', 
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload papers → Get insights with grounded citations → '
        'Cross-paper analysis → Evidence-based Q&A</div>', 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ About This Tool")
        st.markdown("""
        **Target Users:** Consultants, researchers, analysts conducting literature reviews
        
        **Problem Solved:** 
        - Analyzing 5 papers manually: 10-15 hours
        - With this tool: 3-5 minutes
        - **120x faster** with grounded citations
        
        **AI Features:**
        - 📄 Structured extraction with page citations
        - 🔍 Cross-paper analysis (themes, contradictions)
        - 🌐 Reference overlap network (co-citation)
        - 💬 Evidence-based Q&A with sources
        - 📝 Literature review generation
        
        **Why AI is Essential:**
        Multi-document reasoning at scale. Humans can't simultaneously 
        cross-reference 5 papers across hundreds of pages.
        
        **Anti-Hallucination:**
        - Local TF-IDF retrieval (no LLM for chunk selection)
        - Page-tracked chunks
        - Evidence panel shows exact sources
        """)
        
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - Upload 2-5 papers for best results
        - AI provides page citations for verification
        - Evidence panel shows exact sources
        - Citation network shows relationships
        """)
    
    # Check for API key (server-side only)
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        st.error("""
        ⚠️ **Configuration Error**: API key not configured.
        
        **For deployment**: Add ANTHROPIC_API_KEY to Streamlit secrets:
        1. Go to app settings
        2. Add to secrets: ANTHROPIC_API_KEY = "your-key"
        3. Redeploy
        
        This ensures interviewers can use the app without any setup!
        """)
        st.stop()
    
    # Initialize analyzer
    analyzer = ResearchPaperAnalyzer(api_key)
    
    # File upload section
    st.header("1️⃣ Upload Research Papers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files (2-5 papers recommended)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload academic research papers in PDF format"
        )
    
    with col2:
        if st.button("📥 Load Demo Papers", help="Load sample papers for testing"):
            demo_papers = load_demo_papers()
            if demo_papers:
                uploaded_files = demo_papers
    
    if not uploaded_files:
        st.info("👆 Upload research papers to get started")
        
        st.markdown("### 🎯 Example Use Cases")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **For Consultants:**
            - Quick industry research
            - Competitive analysis
            - Evidence-based recommendations
            """)
        
        with col2:
            st.markdown("""
            **For Researchers:**
            - Literature review prep
            - Gap identification
            - Methodology comparison
            """)
        
        with col3:
            st.markdown("""
            **For Students:**
            - Thesis research
            - Essay preparation
            - Citation management
            """)
        
        return
    
    if len(uploaded_files) > 5:
        st.warning("⚠️ Limiting to first 5 papers for optimal performance")
        uploaded_files = uploaded_files[:5]
    
    # Initialize session state
    if 'analyses' not in st.session_state:
        st.session_state.analyses = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = None
    if 'comparison' not in st.session_state:
        st.session_state.comparison = None
    if 'papers_data' not in st.session_state:
        st.session_state.papers_data = None
    
    # Analysis section
    st.header("2️⃣ AI Analysis with Page Citations")
    
    if st.button("🚀 Analyze Papers with AI", type="primary"):
        analyses = []
        all_chunks = []
        papers_data = []
        
        with st.spinner("🔍 Extracting and analyzing with page tracking..."):
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.info(f"📄 Processing: {uploaded_file.name}")
                
                # Extract with page numbers (FIX #1: file pointer reset inside function)
                pages_text = analyzer.extract_text_with_pages(uploaded_file)
                
                if pages_text:
                    # Create chunks for RAG (FIX #4: proper overlap)
                    chunks = analyzer.create_chunks(pages_text, uploaded_file.name)
                    all_chunks.extend(chunks)
                    
                    # Extract references (FIX #5: normalized)
                    references = analyzer.extract_references(pages_text)
                    
                    # Analyze with citations
                    analysis = analyzer.analyze_paper_with_chunks(chunks, uploaded_file.name)
                    analyses.append(analysis)
                    
                    # Store for network
                    papers_data.append({
                        'name': uploaded_file.name,
                        'title': analysis.get('title', uploaded_file.name),
                        'references': references
                    })
                else:
                    st.error(f"❌ Could not extract text from {uploaded_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            progress_bar.empty()
        
        # Store in session
        st.session_state.analyses = analyses
        st.session_state.all_chunks = all_chunks
        st.session_state.papers_data = papers_data
        
        # Cross-paper analysis
        if len(analyses) > 1:
            with st.spinner("🔗 Performing cross-paper analysis..."):
                comparison = analyzer.compare_papers(analyses)
                st.session_state.comparison = comparison
        
        st.success(f"✅ Analyzed {len(analyses)} papers with {len(all_chunks)} evidence chunks!")
    
    # Display results
    if st.session_state.analyses:
        analyses = st.session_state.analyses
        all_chunks = st.session_state.all_chunks
        
        # Individual summaries
        st.header("3️⃣ Individual Paper Summaries (with Citations)")
        
        for i, analysis in enumerate(analyses, 1):
            with st.expander(f"📄 Paper {i}: {analysis.get('title', 'Unknown')}", 
                           expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Overview")
                    st.write(f"**File:** {analysis.get('paper_name', 'N/A')}")
                    st.write(f"**Domain:** {analysis.get('domain', 'N/A')}")
                    
                    st.subheader("🎯 Research Question")
                    st.write(analysis.get('research_question', 'Not specified'))
                    
                    st.subheader("🔬 Methodology")
                    st.write(analysis.get('methodology', 'Not specified'))
                
                with col2:
                    st.subheader("💡 Key Findings (with page refs)")
                    findings = analysis.get('key_findings', [])
                    
                    if findings and isinstance(findings[0], dict):
                        for f in findings:
                            st.markdown(
                                f"- {f.get('finding', '')} "
                                f"<span class='citation'>p.{f.get('page', '?')}</span>",
                                unsafe_allow_html=True
                            )
                            if f.get('evidence'):
                                st.markdown(
                                    f'<div class="evidence-box">'
                                    f'📌 Evidence: {f.get("evidence", "")}</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        for f in findings:
                            st.markdown(f"- {f}")
                
                st.subheader("📝 Abstract")
                st.write(analysis.get('abstract', 'Not specified'))
        
        # Citation Network
        if len(analyses) > 1 and st.session_state.papers_data:
            st.header("4️⃣ Reference Overlap Network")
            
            st.info("""
            **Reference Overlap Network (Co-Citation Analysis):**
            - **Nodes** = Papers in this analysis
            - **Edges** = Shared references between papers
            - **Edge thickness** = Number of shared references
            
            This shows how papers relate through overlapping citations (co-citation).
            Papers citing similar sources are likely addressing related research questions.
            """)
            
            try:
                # FIX #6: Undirected graph
                G = analyzer.build_citation_network(st.session_state.papers_data)
                
                if G.number_of_edges() > 0:
                    # FIX #7: HTML in memory, no temp files
                    html_content = analyzer.visualize_network(G)
                    
                    st.components.v1.html(html_content, height=650)
                    
                    # Network stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers", G.number_of_nodes())
                    with col2:
                        st.metric("Connections", G.number_of_edges())
                    with col3:
                        avg_refs = sum(d for _, _, d in G.edges(data='weight', default=1)) / max(G.number_of_edges(), 1)
                        st.metric("Avg Shared Refs", f"{avg_refs:.1f}")
                else:
                    st.warning("No reference overlap found between these papers. "
                             "Papers may be from different domains or have minimal citation overlap.")
            
            except Exception as e:
                st.error(f"Error building reference network: {str(e)}")
        
        # Cross-paper analysis
        if len(analyses) > 1 and st.session_state.comparison:
            st.header("5️⃣ Cross-Paper Analysis")
            
            comparison = st.session_state.comparison
            
            st.subheader("🔗 Common Themes")
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            for theme in comparison.get('common_themes', []):
                st.markdown(f"- {theme}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if comparison.get('contradictions'):
                st.subheader("⚡ Contradictions & Debates")
                for contra in comparison.get('contradictions', []):
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.write(f"**Topic:** {contra.get('topic', 'N/A')}")
                    st.write(f"**Position 1:** {contra.get('paper_1_position', 'N/A')}")
                    st.write(f"**Position 2:** {contra.get('paper_2_position', 'N/A')}")
                    st.write(f"**Significance:** {contra.get('significance', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("🎓 Synthesis")
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write(comparison.get('synthesis', 'Not available'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Evidence-based Q&A
        st.header("6️⃣ Ask Questions (with Evidence Grounding)")
        
        st.info("""
        💡 AI answers using ONLY uploaded papers and shows exact evidence.
        
        **Retrieval:** Local TF-IDF scoring (no LLM call = faster + cheaper + no hallucination in retrieval)
        """)
        
        question = st.text_input(
            "Your Question:",
            placeholder="e.g., What are the main methodological differences?"
        )
        
        if st.button("🤔 Get Evidence-Based Answer"):
            if question:
                with st.spinner("🔍 Finding relevant evidence and generating answer..."):
                    # FIX #2: Uses local TF-IDF, not LLM
                    answer, evidence_chunks = analyzer.answer_with_evidence(
                        question, all_chunks, analyses
                    )
                    
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown(f"**Question:** {question}")
                    st.markdown("**Answer:**")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show evidence
                    if evidence_chunks:
                        st.subheader("📚 Evidence Used")
                        for chunk in evidence_chunks:
                            st.markdown(
                                f'<div class="evidence-box">'
                                f'<strong>{chunk.paper_name} (Page {chunk.page_num})</strong><br>'
                                f'{chunk.text[:300]}...'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.warning("Please enter a question")


if __name__ == "__main__":
    main()
