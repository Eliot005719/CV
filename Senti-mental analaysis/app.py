
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import io   

from sentiment_analyzer import SentimentAnalyzer
from database_manager import DatabaseManager, CSVExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #0066cc;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_analyzer():
    """Initialize sentiment analyzer (cached for performance)"""
    with st.spinner("Loading AI model (first time only)..."):
        analyzer = SentimentAnalyzer(
            model_name="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return analyzer

@st.cache_resource
def initialize_database(db_type: str = "sqlite"):
    """Initialize database connection (cached)"""
    db = DatabaseManager(db_type="sqlite", db_path="sentiment_analysis.db")
    return db

def plot_sentiment_distribution(stats: dict) -> go.Figure:
    """Create sentiment distribution pie chart"""
    labels = ['Positive', 'Negative', 'Neutral']
    values = [
        stats.get('positive', 0),
        stats.get('negative', 0),
        stats.get('neutral', 0)
    ]
    colors = ['#28a745', '#dc3545', '#ffc107']

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.4)]
    )
    fig.update_layout(
        title="Sentiment Distribution",
        font=dict(size=12),
        height=400,
        showlegend=True
    )
    return fig

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">📊 Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modern NLP-powered sentiment classification with real-time processing</div>', unsafe_allow_html=True)

    # Initialize components
    analyzer = initialize_analyzer()
    db = initialize_database()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write("Database: SQLite (Local)")

        with st.expander("ℹ️ Model Information"):
            st.write("""
            **Model:** DistilBERT (Fine-tuned on SST-2)

            **Why DistilBERT?**
            - 40% faster than BERT
            - 97% of BERT's performance
            - 268MB model size (lightweight)
            - Handles context, sarcasm, and complex language
            """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Single Text", "📁 Batch Upload", "📊 Analytics", "💾 Export"]
    )

    # TAB 1: Single Text Analysis
    with tab1:
        st.header("Analyze Single Text")

        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="E.g., 'This product is amazing! I love it!'",
            height=150
        )

        if st.button("🔍 Analyze", type="primary"):
            if user_text:
                with st.spinner("Analyzing sentiment..."):
                    result = analyzer.analyze_single(user_text)

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sentiment", result.get('sentiment', 'N/A'))

                with col2:
                    st.metric("Confidence", f"{result.get('confidence', 0):.2%}")

                with col3:
                    st.metric("Timestamp", result.get('timestamp', 'N/A'))

                # Detailed scores
                st.subheader("Confidence Scores")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Score", f"{result.get('positive_score', 0):.2%}")
                with col2:
                    st.metric("Negative Score", f"{result.get('negative_score', 0):.2%}")

                # Save to database
                if st.button("💾 Save to Database"):
                    result['model_name'] = 'distilbert'
                    if db.insert_result(result):
                        st.success("✓ Saved to database")
                    else:
                        st.error("✗ Failed to save")

    # TAB 2: Batch Upload
    with tab2:
        st.header("Batch Analysis")

        uploaded_file = st.file_uploader(
            "Upload text file or CSV",
            type=['txt', 'csv'],
            help="TXT: one text per line | CSV: text in 'text' or 'review' column"
        )

        if uploaded_file:
            # Parse file
            if uploaded_file.type == 'text/plain':
                texts = uploaded_file.read().decode('utf-8').split('\n')
                texts = [t.strip() for t in texts if t.strip()]
            else:  # CSV
                df = pd.read_csv(uploaded_file)
                # Find text column
                text_col = None
                for col in ['text', 'review', 'content', 'message']:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col:
                    texts = df[text_col].dropna().tolist()
                else:
                    st.error("CSV must contain 'text', 'review', 'content', or 'message' column")
                    texts = []

            if texts:
                st.info(f"📄 Loaded {len(texts)} texts for analysis")

                if st.button("🚀 Analyze All", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("Analyzing..."):
                        results = []
                        for idx, text in enumerate(texts):
                            result = analyzer.analyze_single(text)
                            result['model_name'] = 'distilbert'
                            results.append(result)

                            progress = (idx + 1) / len(texts)
                            progress_bar.progress(progress)
                            status_text.text(f"Progress: {idx + 1}/{len(texts)}")

                    # Save to database
                    inserted = db.insert_batch(results)
                    st.success(f"✓ Analysis complete! {inserted} records saved to database")

                    # Display statistics
                    stats = analyzer.get_statistics(results)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", stats['total'])
                    with col2:
                        st.metric("Positive", f"{stats['positive']} ({stats['positive_pct']}%)")
                    with col3:
                        st.metric("Negative", f"{stats['negative']} ({stats['negative_pct']}%)")
                    with col4:
                        st.metric("Neutral", f"{stats['neutral']} ({stats['neutral_pct']}%)")

                    # Results table
                    st.subheader("Detailed Results")
                    results_df = pd.DataFrame([
                        {
                            'Text': r.get('text', '')[:50] + '...',
                            'Sentiment': r.get('sentiment', 'N/A'),
                            'Confidence': f"{r.get('confidence', 0):.2%}",
                        }
                        for r in results
                    ])
                    st.dataframe(results_df, use_container_width=True)

    # TAB 3: Analytics
    with tab3:
        st.header("Analytics Dashboard")

        all_results = db.get_all_results(limit=1000)

        if all_results:
            stats = analyzer.get_statistics(all_results)

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", stats['total'])
            with col2:
                st.metric("Positive", f"{stats['positive']} ({stats['positive_pct']}%)")
            with col3:
                st.metric("Negative", f"{stats['negative']} ({stats['negative_pct']}%)")
            with col4:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = plot_sentiment_distribution(stats)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data in database. Start analyzing texts to populate the dashboard.")

    # TAB 4: Export
    with tab4:
        st.header("Export Results")

        all_results = db.get_all_results()

        if all_results:
            st.success(f"✓ {len(all_results)} records available for export")

            if st.button("📊 Export All as CSV"):
                export_df = pd.DataFrame([
                    {
                        'Original Text': r.get('original_text', r.get('text', ''))[:100],
                        'Sentiment': r.get('sentiment', 'N/A'),
                        'Confidence': r.get('confidence', 0),
                        'Positive Score': r.get('positive_score', 0),
                        'Negative Score': r.get('negative_score', 0),
                        'Timestamp': r.get('created_at', r.get('timestamp', ''))
                    }
                    for r in all_results
                ])

                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No data to export. Start analyzing texts first.")

if __name__ == "__main__":
    main()
