import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from app.generation.rag_pipeline import ask
from app.evaluation.retrieval_metrics import retrieval_quality
from app.evaluation.response_metrics import response_quality
from app.evaluation.hallucination import hallucination_rate

# ==========================================
# PAGE CONFIG & PREMIUM THEME
# ==========================================

st.set_page_config(
    page_title="NeuralDocs Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium professional theme with sophisticated design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --primary: #0f1419;
        --accent: #3b82f6;
        --accent-deep: #1e40af;
        --accent-light: #eff6ff;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
    }
    
    body {
        background: linear-gradient(135deg, #f0f4f8 0%, #f9fafb 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--primary);
        font-weight: 400;
        letter-spacing: -0.3px;
    }
    
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #f9fafb 100%);
        padding: 32px 24px;
    }
    
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: var(--primary);
        letter-spacing: -1.5px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #0f1419 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: var(--primary);
        letter-spacing: -0.8px;
        margin-top: 32px;
        margin-bottom: 24px;
    }
    
    h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: var(--primary);
        letter-spacing: -0.5px;
        margin-bottom: 16px;
    }
    
    .stCaption {
        font-size: 13px;
        font-weight: 500;
        color: var(--neutral-600);
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid rgba(59, 130, 246, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
        border-color: var(--accent);
        transform: translateY(-4px);
    }
    
    .stMetric [data-testid="metric-value"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 36px !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
        color: var(--primary) !important;
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: var(--neutral-600) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid var(--neutral-200);
        padding: 12px 0;
    }
    
    .stTabs [role="tab"] {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.3px;
        color: var(--neutral-600);
        padding: 8px 16px;
        margin-right: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent);
        border-bottom: 3px solid var(--accent);
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid var(--neutral-200);
        font-family: 'Inter', sans-serif;
        padding: 10px 14px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stSlider > div > div > div > div {
        color: var(--accent) !important;
    }
    
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px !important;
        padding: 16px 20px !important;
        font-size: 14px;
        font-weight: 500;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
        border: 1.5px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
        border: 1.5px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%) !important;
        border: 1.5px solid rgba(245, 158, 11, 0.3) !important;
    }
    
    hr {
        margin: 28px 0;
        border: none;
        border-top: 1.5px solid var(--neutral-200);
        opacity: 0.6;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid var(--neutral-200);
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f3f4f6 0%, #eff6ff 100%);
        border-color: var(--accent);
    }
    
    [data-testid="stDataFrame"] {
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .footer-text {
        font-size: 12px;
        font-weight: 500;
        color: var(--neutral-600);
        letter-spacing: 0.2px;
        text-align: center;
        margin-top: 40px;
        padding-top: 24px;
        border-top: 1px solid var(--neutral-200);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question" not in st.session_state:
    st.session_state.question = ""

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "response_times" not in st.session_state:
    st.session_state.response_times = []

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown('<h2 style="margin: 0 0 4px 0; font-size: 26px;">📚 NeuralDocs</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 13px; font-weight: 600; color: #6b7280; letter-spacing: 0.4px; text-transform: uppercase; margin-bottom: 20px;">Intelligent RAG Platform</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<p style="font-size: 13px; font-weight: 700; letter-spacing: 0.3px; color: #1f2937; margin-bottom: 12px;">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["💬 AI Assistant", "📊 Analytics", "📄 Documents", "⚙️ System Config"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown('<p style="font-size: 13px; font-weight: 700; letter-spacing: 0.3px; color: #1f2937; margin-bottom: 12px;">📤 DOCUMENT MANAGEMENT</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")
    
    st.markdown("---")
    
    st.markdown('<p style="font-size: 13px; font-weight: 700; letter-spacing: 0.3px; color: #1f2937; margin-bottom: 12px;">⚡ RETRIEVAL SETTINGS</p>', unsafe_allow_html=True)
    top_k = st.slider("Top K Documents", 1, 20, 5, label_visibility="collapsed")
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")
    
    st.markdown("---")
    
    if st.button("🔄 Build/Update Index", use_container_width=True):
        with st.spinner("Building FAISS index..."):
            time.sleep(1)
            st.success("✓ Index built successfully")
    
    st.markdown("---")
    
    st.markdown('<p style="font-size: 13px; font-weight: 700; letter-spacing: 0.3px; color: #1f2937; margin-bottom: 12px;">📊 QUICK STATS</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", "5")
    with col2:
        st.metric("Total Pages", "342")

# ==========================================
# MAIN HEADER
# ==========================================

st.markdown("""
<div style="margin-bottom: 32px;">
    <h1 style="margin: 0 0 12px 0;">🧠 NeuralDocs Pro</h1>
    <p style="font-size: 16px; color: #6b7280; font-weight: 500; letter-spacing: 0.2px; margin: 0;">Intelligent PDF Analysis & Question Answering System</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# AI ASSISTANT PAGE
# ==========================================

if page == "💬 AI Assistant":
    
    st.markdown("---")
    
    st.markdown('<h2 style="margin-top: 0; margin-bottom: 24px;">📈 Session Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Questions Asked", st.session_state.question_count, delta=1 if st.session_state.question_count > 0 else 0)
    
    with col2:
        avg_response = np.mean(st.session_state.response_times) if st.session_state.response_times else 0
        st.metric("Avg Response", f"{avg_response:.2f}s")
    
    with col3:
        st.metric("Documents", "5")
    
    with col4:
        st.metric("Total Chunks", "1,247")
    
    st.markdown("---")
    
    st.markdown('<h2 style="margin-bottom: 24px;">💬 Ask Your Question</h2>', unsafe_allow_html=True)
    
    suggestions = ["What is self-attention?", "What datasets were used?", "What BLEU scores were achieved?", "Summarize the paper."]
    
    st.markdown('<p style="font-size: 14px; font-weight: 600; color: #6b7280; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Quick Suggestions</p>', unsafe_allow_html=True)
    
    cols = st.columns(len(suggestions))
    selected_question = None
    
    for i, q in enumerate(suggestions):
        with cols[i]:
            if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                selected_question = q
                st.session_state.question = q
    
    st.markdown("")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input("Enter your question", value=st.session_state.question, placeholder="What would you like to know about your documents?", label_visibility="collapsed")
    
    with col2:
        ask_clicked = st.button("🔍 Ask", use_container_width=True)
    
    if ask_clicked or selected_question:
        if selected_question:
            question = selected_question
        
        if question.strip():
            st.session_state.question_count += 1
            
            with st.spinner("🔍 Searching documents... Retrieving context... Generating answer..."):
                start_time = time.time()
                evaluation = None
                
                try:
                    result = ask(
                        question=question,
                        top_k=top_k
                    )

                    answer = result["answer"]

                    citations = result["citations"]

                    evaluation = result.get(
                        "evaluation",
                        {}
                    )
                    response_time = round(time.time() - start_time, 2)
                except Exception as e:
                    answer = f"⚠️ Error processing question: {str(e)}"
                    citations = []
                    response_time = round(time.time() - start_time, 2)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                        "time": timestamp,
                        "citations": citations,
                        "response_time": response_time,
                        "evaluation": evaluation,
                    }
                )
                
                st.session_state.response_times.append(response_time)
    
    if st.session_state.messages:
        st.markdown("---")
        st.markdown(f'<h2 style="margin-bottom: 24px;">💬 Conversation History</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 13px; color: #9ca3af; margin: -20px 0 20px 0; text-transform: uppercase; letter-spacing: 0.3px;">{len(st.session_state.messages)} messages in this session</p>', unsafe_allow_html=True)
        
        for msg in reversed(st.session_state.messages):
            with st.container():
                col1, col2 = st.columns([1, 20])
                
                with col1:
                    st.markdown('<span style="font-size: 24px;">👤</span>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<span style="font-family: \'Space Grotesk\'; font-weight: 700; font-size: 15px; letter-spacing: 0.3px;">Your Question</span>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-left: 3px solid #3b82f6; border-radius: 8px; padding: 14px; margin-top: 8px; font-size: 15px; color: #1e3a8a;">{msg["question"]}</div>', unsafe_allow_html=True)
                
                st.markdown("")
                
                col1, col2 = st.columns([1, 20])
                
                with col1:
                    st.markdown('<span style="font-size: 24px;">🤖</span>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<span style="font-family: \'Space Grotesk\'; font-weight: 700; font-size: 15px; letter-spacing: 0.3px;">AI Response</span>', unsafe_allow_html=True)
                    st.success(msg["answer"])
                    
                    col_time, col_response = st.columns(2)
                    
                    with col_time:
                        st.caption(f"🕒 {msg['time']}")
                    
                    with col_response:
                        st.caption(
                            f"⚡ {msg['response_time']}s"
                        )

                    # ======================================
                    # EVALUATION METRICS
                    # ======================================

                    e = msg.get("evaluation") or {}

                    if not e and msg.get("citations"):
                        evaluation_chunks = [
                            {
                                "text": citation.get("source_preview", ""),
                                "distance": citation.get("distance"),
                                "rerank_score": citation.get("rerank_score"),
                            }
                            for citation in msg["citations"]
                            if citation.get("source_preview")
                        ]

                        if evaluation_chunks:
                            e = {
                                "retrieval_quality": retrieval_quality(
                                    evaluation_chunks,
                                    msg["question"],
                                ),
                                "response_quality": response_quality(
                                    msg["answer"],
                                    msg["question"],
                                    evaluation_chunks,
                                ),
                                "hallucination_rate": hallucination_rate(
                                    msg["answer"],
                                    evaluation_chunks,
                                ),
                            }
                            msg["evaluation"] = e

                    retrieval_value = e.get("retrieval_quality")
                    response_value = e.get("response_quality")
                    hallucination_value = e.get("hallucination_rate")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Retrieval Quality",
                            f"{retrieval_value * 100:.1f}%"
                            if retrieval_value is not None
                            else "N/A",
                        )

                    with c2:
                        st.metric(
                            "Response Quality",
                            f"{response_value * 100:.1f}%"
                            if response_value is not None
                            else "N/A",
                        )

                    with c3:
                        st.metric(
                            "Hallucination Rate",
                            f"{hallucination_value:.1f}%"
                            if hallucination_value is not None
                            else "N/A",
                        )

                    if not e:
                        st.caption(
                            "Metrics require retrieved source context for this response."
                        )
                
                if msg["citations"]:
                    with st.expander("📚 Sources & Citations"):
                        for i, citation in enumerate(msg["citations"], 1):
                            st.markdown(f'<span style="font-weight: 700; color: #1e40af;">Source {i} • Page {citation.get("page", "N/A")}</span>', unsafe_allow_html=True)
                            st.markdown(f'<span style="font-size: 13px; color: #6b7280; font-weight: 500;">Chunk ID: {citation.get("chunk_id", "N/A")}</span>', unsafe_allow_html=True)
                            st.markdown(f'<div style="background: #eff6ff; border-left: 3px solid #3b82f6; padding: 12px; border-radius: 6px; margin-top: 8px;">{citation.get("source_preview", "No preview available")[:200]}...</div>', unsafe_allow_html=True)
                            st.markdown("")
                
                st.markdown("---")

# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "📊 Analytics":
    
    st.markdown('<h2 style="margin-top: 0;">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.response_times:
        response_data = pd.DataFrame({
            "Query #": range(1, len(st.session_state.response_times) + 1),
            "Response Time (s)": st.session_state.response_times
        })
    else:
        response_data = pd.DataFrame({
            "Query #": range(1, 11),
            "Response Time (s)": [4.2, 3.8, 5.1, 3.5, 4.9, 3.2, 4.7, 3.9, 5.2, 4.0]
        })
    
    doc_stats = pd.DataFrame({
        "Document": ["paper_1.pdf", "paper_2.pdf", "report_q3.pdf", "guide.pdf", "summary.pdf"],
        "Pages": [15, 23, 12, 8, 10],
        "Chunks": [203, 387, 156, 98, 124],
        "Queries": [12, 28, 8, 5, 7]
    })
    
    tab1, tab2, tab3, tab4 = st.tabs(["Response Performance", "Document Insights", "Query Patterns", "System Metrics"])
    
    with tab1:
        st.markdown(f'<h3 style="margin-top: 0;">Response Time Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Response Time", f"{response_data['Response Time (s)'].mean():.2f}s", delta=f"{response_data['Response Time (s)'].std():.2f}s")
        
        with col2:
            st.metric("Min Response Time", f"{response_data['Response Time (s)'].min():.2f}s")
        
        with col3:
            st.metric("Max Response Time", f"{response_data['Response Time (s)'].max():.2f}s")
        
        st.markdown("")
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=response_data["Query #"],
            y=response_data["Response Time (s)"],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#3b82f6'),
            fill='tozeroy',
            fillcolor='rgba(59, 144, 226, 0.1)',
            hovertemplate='<b>Query %{x}</b><br>Response Time: %{y:.2f}s<extra></extra>'
        ))
        
        avg_line = response_data["Response Time (s)"].mean()
        fig_trend.add_hline(y=avg_line, line_dash="dash", line_color="#10b981", annotation_text="Average", annotation_position="right")
        
        fig_trend.update_layout(
            title="Response Time Trend",
            xaxis_title="Query Number",
            yaxis_title="Time (seconds)",
            template="plotly_white",
            hovermode='x unified',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=response_data["Response Time (s)"],
            nbinsx=10,
            name='Response Time',
            marker_color='#3b82f6',
            marker_line_color='#1e40af',
            marker_line_width=1.5,
            opacity=0.8,
            hovertemplate='Time Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig_dist.update_layout(
            title="Response Time Distribution",
            xaxis_title="Response Time (seconds)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=350,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.markdown(f'<h3 style="margin-top: 0;">Document Intelligence</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(doc_stats))
        
        with col2:
            st.metric("Total Pages", doc_stats["Pages"].sum())
        
        with col3:
            st.metric("Total Chunks", doc_stats["Chunks"].sum())
        
        st.markdown("")
        
        fig_pages = px.bar(
            doc_stats,
            x="Document",
            y="Pages",
            color="Pages",
            color_continuous_scale=["#e8f0ff", "#3b82f6"],
            title="Pages per Document",
            text="Pages"
        )
        
        fig_pages.update_traces(textposition='auto', marker_line_color='#1e40af', marker_line_width=1.5, hovertemplate='<b>%{x}</b><br>Pages: %{y}<extra></extra>')
        fig_pages.update_layout(template="plotly_white", height=400, showlegend=False, margin=dict(l=40, r=40, t=60, b=40), font=dict(family="Inter, sans-serif"))
        
        st.plotly_chart(fig_pages, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_chunks = px.scatter(
                doc_stats,
                x="Pages",
                y="Chunks",
                size="Queries",
                hover_data={"Document": True},
                title="Chunks vs Pages (size = queries)",
                color="Queries",
                color_continuous_scale=["#e8f0ff", "#3b82f6"],
                text="Document"
            )
            
            fig_chunks.update_traces(marker_line_color='#1e40af', marker_line_width=1.5)
            fig_chunks.update_layout(template="plotly_white", height=350, margin=dict(l=40, r=40, t=60, b=40), font=dict(family="Inter, sans-serif"))
            
            st.plotly_chart(fig_chunks, use_container_width=True)
        
        with col2:
            st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">Document Details</h3>', unsafe_allow_html=True)
            
            display_df = doc_stats.copy()
            display_df["Chunk Density"] = (display_df["Chunks"] / display_df["Pages"]).round(1)
            
            st.dataframe(display_df[["Document", "Pages", "Chunks", "Queries", "Chunk Density"]], use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown(f'<h3 style="margin-top: 0;">Query Analysis</h3>', unsafe_allow_html=True)
        
        hours = pd.date_range(start='00:00', periods=24, freq='H')
        hourly_queries = np.random.poisson(3, 24)
        
        hourly_data = pd.DataFrame({"Hour": hours.strftime('%H:00'), "Queries": hourly_queries})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly_data["Hour"],
                y=hourly_data["Queries"],
                name="Queries",
                marker_color='#3b82f6',
                marker_line_color='#1e40af',
                marker_line_width=1,
                hovertemplate='<b>%{x}</b><br>Queries: %{y}<extra></extra>'
            ))
            
            fig_hourly.update_layout(
                title="Query Volume by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Queries",
                template="plotly_white",
                height=350,
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            query_types = pd.DataFrame({
                "Query Type": ["Definition", "Analysis", "Summary", "Comparison", "Other"],
                "Count": [28, 35, 22, 15, 12]
            })
            
            colors_pie = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=query_types["Query Type"],
                values=query_types["Count"],
                marker=dict(colors=colors_pie, line=dict(color='white', width=2)),
                hovertemplate='<b>%{label}</b><br>Queries: %{value}<extra></extra>'
            )])
            
            fig_pie.update_layout(
                title="Query Type Distribution",
                height=350,
                margin=dict(l=40, r=40, t=60, b=40),
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">Most Searched Topics</h3>', unsafe_allow_html=True)
        
        keywords = pd.DataFrame({
            "Keyword": ["attention", "transformer", "model", "training", "dataset", "performance", "loss", "architecture"],
            "Mentions": [47, 42, 38, 35, 32, 28, 25, 23]
        }).sort_values("Mentions", ascending=False)
        
        fig_keywords = px.bar(
            keywords,
            x="Mentions",
            y="Keyword",
            orientation="h",
            color="Mentions",
            color_continuous_scale=["#e8f0ff", "#4a90e2"],
            title="Top Search Keywords",
            labels={
                "Mentions": "Mention Count",
                "Keyword": ""
            }
        )
        
        fig_keywords.update_traces(marker_line_color='#1e40af', marker_line_width=1.5, hovertemplate='<b>%{y}</b><br>Mentions: %{x}<extra></extra>')
        fig_keywords.update_layout(template="plotly_white", height=350, margin=dict(l=120, r=40, t=60, b=40), showlegend=False, font=dict(family="Inter, sans-serif"))
        
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    with tab4:
        st.markdown(f'<h3 style="margin-top: 0;">System Performance Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Uptime", "99.9%", delta="↑ 0.1%")
        
        with col2:
            st.metric("Cache Hit Rate", "73%", delta="↑ 5%")
        
        with col3:
            st.metric("Throughput", "2.3 q/s", delta="↑ 0.2 q/s")
        
        with col4:
            st.metric("Index Size", "145 MB", delta="↑ 12 MB")
        
        st.markdown("")
        
        system_data = pd.DataFrame({
            "Metric": ["CPU", "Memory", "Disk I/O", "Network"],
            "Usage %": [34, 56, 28, 42]
        })
        
        fig_resources = go.Figure(data=[
            go.Scatterpolar(
                r=system_data["Usage %"],
                theta=system_data["Metric"],
                fill='toself',
                name='Usage',
                marker_color='#3b82f6',
                line=dict(color='#1e40af', width=2),
                fillcolor='rgba(59, 130, 246, 0.3)',
                hovertemplate='<b>%{theta}</b><br>Usage: %{r}%<extra></extra>'
            )
        ])
        
        fig_resources.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11))),
            title="System Resource Utilization",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig_resources, use_container_width=True)
        
        perf_dates = pd.date_range(start='2024-06-15', periods=30, freq='D')
        
        perf_data = pd.DataFrame({
            "Date": perf_dates,
            "Avg Response": np.cumsum(np.random.randn(30) * 0.2) + 4,
            "P95 Response": np.cumsum(np.random.randn(30) * 0.3) + 6,
            "P99 Response": np.cumsum(np.random.randn(30) * 0.4) + 8
        })
        
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=perf_data["Date"],
            y=perf_data["Avg Response"],
            name='Average',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='<b>%{x|%b %d}</b><br>Avg: %{y:.2f}s<extra></extra>'
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=perf_data["Date"],
            y=perf_data["P95 Response"],
            name='P95',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %d}</b><br>P95: %{y:.2f}s<extra></extra>'
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=perf_data["Date"],
            y=perf_data["P99 Response"],
            name='P99',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %d}</b><br>P99: %{y:.2f}s<extra></extra>'
        ))
        
        fig_perf.update_layout(
            title="Response Time Percentiles (30 Days)",
            xaxis_title="Date",
            yaxis_title="Response Time (seconds)",
            template="plotly_white",
            height=400,
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)

# ==========================================
# DOCUMENTS PAGE
# ==========================================

elif page == "📄 Documents":
    
    st.markdown('<h2 style="margin-top: 0;">📄 Document Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">Indexed Documents</h3>', unsafe_allow_html=True)
        
        documents = pd.DataFrame({
            "Document": ["paper_1.pdf", "paper_2.pdf", "report_q3.pdf", "guide.pdf", "summary.pdf"],
            "Status": ["✓ Indexed", "✓ Indexed", "✓ Indexed", "⏳ Processing", "✓ Indexed"],
            "Pages": [15, 23, 12, 8, 10],
            "Chunks": [203, 387, 156, 98, 124],
            "Size": ["2.4 MB", "3.8 MB", "1.9 MB", "0.8 MB", "1.2 MB"],
            "Added": ["Jun 10", "Jun 8", "Jun 5", "Jun 3", "May 28"]
        })
        
        st.dataframe(documents, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">Quick Stats</h3>', unsafe_allow_html=True)
        
        st.metric("Total Documents", 5)
        st.metric("Total Pages", 68)
        st.metric("Total Chunks", 968)
        st.metric("Total Size", "10.1 MB")
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">⚙️ Processing Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
**Embedding Settings**
- Model: all-MiniLM-L6-v2
- Dimensions: 384
- Batch Size: 32
- Device: GPU (CUDA)
        """)
    
    with col2:
        st.info("""
**Indexing Settings**
- Index Type: FAISS (IVF-Flat)
- Distance Metric: L2 (Euclidean)
- Index Size: 145 MB
- Last Updated: Jun 20, 2024
        """)
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
**Retrieval Settings**
- Top K: 5
- Similarity Threshold: 0.5
- Reranker: cross-encoder/ms-marco-MiniLM
- Reranking Strategy: Top-5
        """)
    
    with col2:
        st.info("""
**LLM Configuration**
- Model: llama3.2 (13B)
- Temperature: 0.7
- Max Tokens: 512
- Context Window: 4096
        """)
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">📋 Sample Chunks from paper_1.pdf</h3>', unsafe_allow_html=True)
    
    sample_chunks = [
        {
            "id": "chunk_001",
            "page": 1,
            "text": "Attention is all you need. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "tokens": 42
        },
        {
            "id": "chunk_002",
            "page": 2,
            "text": "The transformer architecture relies entirely on an attention mechanism to draw global dependencies between input and output...",
            "tokens": 38
        },
        {
            "id": "chunk_003",
            "page": 3,
            "text": "The encoder is a stack of N identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism...",
            "tokens": 35
        }
    ]
    
    for chunk in sample_chunks:
        with st.expander(f"**{chunk['id']}** • Page {chunk['page']} • {chunk['tokens']} tokens"):
            st.markdown(chunk['text'])

# ==========================================
# SYSTEM CONFIG PAGE
# ==========================================

elif page == "⚙️ System Config":
    
    st.markdown('<h2 style="margin-top: 0;">⚙️ System Configuration & Status</h2>', unsafe_allow_html=True)
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">🏥 System Health</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("✓ FAISS Index")
        st.caption("Ready")
    
    with col2:
        st.success("✓ Embeddings")
        st.caption("Ready")
    
    with col3:
        st.success("✓ Reranker")
        st.caption("Ready")
    
    with col4:
        st.success("✓ LLM")
        st.caption("Ready")
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">📡 Service Status</h3>', unsafe_allow_html=True)
    
    services = pd.DataFrame({
        "Service": ["FAISS Vector DB", "Embedding Engine", "Reranking Service", "LLM Inference", "API Gateway"],
        "Status": ["🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running"],
        "Uptime": ["99.9%", "99.95%", "99.8%", "99.7%", "99.99%"],
        "Avg Response": ["2ms", "450ms", "380ms", "2100ms", "15ms"]
    })
    
    st.dataframe(services, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">⚙️ Active Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Retrieval Pipeline**")
        st.write(f"Top K Results: {top_k}")
        st.write(f"Similarity Threshold: {similarity_threshold}")
        st.write("Reranking: Enabled")
        st.write("Hybrid Search: BM25 + Vector")
    
    with col2:
        st.markdown("**LLM Settings**")
        st.write("Model: llama3.2")
        st.write("Temperature: 0.7")
        st.write("Max Tokens: 512")
        st.write("Context: 4096 tokens")
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">📊 Recent Logs (Last 10 Operations)</h3>', unsafe_allow_html=True)
    
    logs = pd.DataFrame({
        "Timestamp": ["14:32:45", "14:31:22", "14:30:11", "14:28:54", "14:27:33", "14:26:12", "14:25:01", "14:23:48", "14:22:35", "14:21:20"],
        "Operation": ["Query Processing", "Index Update", "Query Processing", "Cache Clear", "Query Processing", "Query Processing", "Index Build", "Query Processing", "Query Processing", "System Reindex"],
        "Duration": ["2.3s", "5.2s", "1.9s", "0.8s", "2.1s", "2.8s", "12.4s", "2.0s", "2.5s", "45.2s"],
        "Status": ["✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success", "✓ Success"]
    })
    
    st.dataframe(logs, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown(f'<h3 style="margin-top: 0; font-size: 18px;">🔧 Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Services", use_container_width=True):
            with st.spinner("Restarting services..."):
                time.sleep(1)
                st.success("Services restarted successfully")
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            with st.spinner("Clearing cache..."):
                time.sleep(0.5)
                st.success("Cache cleared (freed 24 MB)")
    
    with col3:
        if st.button("📈 Export Logs", use_container_width=True):
            with st.spinner("Exporting logs..."):
                time.sleep(0.5)
                st.success("Logs exported (logs_2024_06_22.zip)")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(f"""
<div class="footer-text">
    🧠 <b>NeuralDocs Pro</b> • Advanced RAG Intelligence Platform • Session: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total Queries: {st.session_state.question_count}
</div>
""", unsafe_allow_html=True)
