import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="🎭",
    layout="wide"
)

# Title and description
st.title("🎭 Sentiment Analysis Tool")
st.markdown("**Analyze emotions in text using Natural Language Processing**")
st.markdown("---")

# Helper function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
        emoji = "😊"
        color = "#2ecc71"
    elif polarity < -0.1:
        sentiment = "Negative"
        emoji = "😞"
        color = "#e74c3c"
    else:
        sentiment = "Neutral"
        emoji = "😐"
        color = "#95a5a6"
    
    return {
        "text": text,
        "sentiment": sentiment,
        "polarity": polarity,
        "emoji": emoji,
        "color": color
    }

# Sidebar for navigation
st.sidebar.title("📊 Analysis Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type:",
    ["Single Text Analysis", "Bulk Analysis (CSV)"]
)

# Single Text Analysis
if analysis_type == "Single Text Analysis":
    st.subheader("✍️ Analyze Individual Text")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here... (e.g., product reviews, tweets, feedback)"
    )
    
    # Example texts
    st.markdown("**Try these examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Positive Example"):
            user_input = "This product is absolutely amazing! Best purchase I've ever made. Highly recommend!"
            
    with col2:
        if st.button("😞 Negative Example"):
            user_input = "Terrible experience. Waste of money. Very disappointed with the quality."
            
    with col3:
        if st.button("😐 Neutral Example"):
            user_input = "The product is okay. Nothing special, but it works as described."
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            result = analyze_sentiment(user_input)
            
            st.markdown("---")
            st.subheader("📈 Analysis Results")
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {result['emoji']} Sentiment")
                st.markdown(f"<h2 style='color: {result['color']};'>{result['sentiment']}</h2>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Polarity Score")
                st.markdown(f"<h2>{result['polarity']:.2f}</h2>", unsafe_allow_html=True)
                st.caption("Range: -1 (negative) to +1 (positive)")
            
            with col3:
                st.markdown("### 🎯 Confidence")
                confidence = abs(result['polarity']) * 100
                st.markdown(f"<h2>{confidence:.1f}%</h2>", unsafe_allow_html=True)
                st.caption("How strong the sentiment is")
            
            # Gauge chart for polarity
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['polarity'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': result['color']},
                    'steps': [
                        {'range': [-1, -0.1], 'color': "#ffcccc"},
                        {'range': [-0.1, 0.1], 'color': "#e0e0e0"},
                        {'range': [0.1, 1], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result['polarity']
                    }
                },
                title = {'text': "Sentiment Meter"}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Please enter some text to analyze!")

# Bulk Analysis
else:
    st.subheader("📁 Bulk Sentiment Analysis")
    st.markdown("Upload a CSV file with a column containing text data for analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
        
        # Show preview
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
        
        # Select column to analyze
        text_column = st.selectbox(
            "Select the column containing text to analyze:",
            df.columns.tolist()
        )
        
        if st.button("🚀 Analyze All Texts", type="primary"):
            with st.spinner("Analyzing sentiments... Please wait."):
                # Analyze all texts
                results = []
                for text in df[text_column]:
                    if pd.notna(text):
                        result = analyze_sentiment(str(text))
                        results.append(result)
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                st.markdown("---")
                st.subheader("📊 Bulk Analysis Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                sentiment_counts = results_df['sentiment'].value_counts()
                
                with col1:
                    st.metric("Total Analyzed", len(results_df))
                
                with col2:
                    positive_count = sentiment_counts.get('Positive', 0)
                    positive_pct = (positive_count / len(results_df)) * 100
                    st.metric("😊 Positive", f"{positive_count} ({positive_pct:.1f}%)")
                
                with col3:
                    neutral_count = sentiment_counts.get('Neutral', 0)
                    neutral_pct = (neutral_count / len(results_df)) * 100
                    st.metric("😐 Neutral", f"{neutral_count} ({neutral_pct:.1f}%)")
                
                with col4:
                    negative_count = sentiment_counts.get('Negative', 0)
                    negative_pct = (negative_count / len(results_df)) * 100
                    st.metric("😞 Negative", f"{negative_count} ({negative_pct:.1f}%)")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        sentiment_counts,
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#95a5a6',
                            'Negative': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        title="Sentiment Count",
                        labels={'x': 'Sentiment', 'y': 'Count'},
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#95a5a6',
                            'Negative': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Polarity distribution
                fig_hist = px.histogram(
                    results_df,
                    x='polarity',
                    nbins=30,
                    title="Polarity Score Distribution",
                    labels={'polarity': 'Polarity Score'},
                    color_discrete_sequence=['#3498db']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                display_df = results_df[['text', 'sentiment', 'polarity', 'emoji']].copy()
                display_df.columns = ['Text', 'Sentiment', 'Polarity Score', 'Emoji']
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Upload a CSV file to get started with bulk analysis!")
        
        # Sample CSV format
        st.markdown("### 📝 Sample CSV Format")
        st.markdown("Your CSV should have at least one column with text data. Example:")
        
        sample_data = {
            'review': [
                'This product is amazing!',
                'Not satisfied with the quality',
                'Average product, nothing special'
            ],
            'rating': [5, 2, 3]
        }
        st.dataframe(pd.DataFrame(sample_data))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit & TextBlob | NLP Sentiment Analysis Tool</p>
    <p>💡 Tip: The polarity score ranges from -1 (most negative) to +1 (most positive)</p>
</div>

""", unsafe_allow_html=True)

