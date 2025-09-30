pip install textblob
python -m textblob.download_corpora

import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="wide")

# Title
st.title("😊 Sentiment Analysis Tool")
st.markdown("### Analyze the sentiment of your text instantly")
st.markdown("---")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This tool analyzes text sentiment:\n\n"
        "😊 **Positive** - Happy, good vibes\n\n"
        "😐 **Neutral** - Neither good nor bad\n\n"
        "😞 **Negative** - Sad, angry, bad vibes"
    )

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "Positive", polarity, "😊"
    elif polarity < 0:
        return "Negative", polarity, "😞"
    else:
        return "Neutral", polarity, "😐"

# Main tabs
tab1, tab2 = st.tabs(["📝 Analyze Text", "📊 View Results"])

# Tab 1: Analyze Text
with tab1:
    st.header("Enter Your Text")

    # Single text input
    user_input = st.text_area(
        "Type or paste text here:",
        height=150,
        placeholder="e.g., I love this product! It's amazing!"
    )

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if user_input.strip():
            sentiment, polarity, emoji = analyze_sentiment(user_input)

            # Show results
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", f"{emoji} {sentiment}")
            col2.metric("Polarity Score", f"{polarity:.2f}")
            col3.metric("Confidence", f"{abs(polarity) * 100:.0f}%")

            # Visual indicator
            if sentiment == "Positive":
                st.success(f"✅ This text is {sentiment}!")
            elif sentiment == "Negative":
                st.error(f"❌ This text is {sentiment}!")
            else:
                st.info(f"ℹ️ This text is {sentiment}!")

            # Save to results
            st.session_state.results.append({
                'Text': user_input[:50] + '...' if len(user_input) > 50 else user_input,
                'Sentiment': sentiment,
                'Polarity': polarity
            })
        else:
            st.warning("⚠️ Please enter some text!")

    st.markdown("---")

    # Multiple texts input
    st.subheader("Analyze Multiple Texts")
    multi_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=150,
        placeholder="Text 1\nText 2\nText 3..."
    )

    if st.button("🔍 Analyze All", type="primary", use_container_width=True, key="multi"):
        if multi_text.strip():
            texts = [t.strip() for t in multi_text.split('\n') if t.strip()]
            for text in texts:
                sentiment, polarity, emoji = analyze_sentiment(text)
                st.session_state.results.append({
                    'Text': text[:50] + '...' if len(text) > 50 else text,
                    'Sentiment': sentiment,
                    'Polarity': polarity
                })
            st.success(f"✅ Analyzed {len(texts)} texts!")
        else:
            st.warning("⚠️ Please enter some texts!")

# Tab 2: View Results
with tab2:
    st.header("Analysis Results")

    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("😊 Positive", len(df[df['Sentiment'] == 'Positive']))
        col2.metric("😐 Neutral", len(df[df['Sentiment'] == 'Neutral']))
        col3.metric("😞 Negative", len(df[df['Sentiment'] == 'Negative']))

        st.markdown("---")

        # Show table
        st.subheader("📋 All Results")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # Visualization
        col1, col2 = st.columns(2)
        sentiment_counts = df['Sentiment'].value_counts()

        with col1:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': 'lightgreen',
                    'Negative': 'lightcoral',
                    'Neutral': 'lightgray'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Sentiment Count")
            fig_bar = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                labels={'x': 'Sentiment', 'y': 'Count'},
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': 'lightgreen',
                    'Negative': 'lightcoral',
                    'Neutral': 'lightgray'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Download and clear buttons
        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear All Results", type="secondary", use_container_width=True):
                st.session_state.results = []
                st.success("All results cleared!")
                st.experimental_rerun()
    else:
        st.info("📊 No results yet. Analyze some text first!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & TextBlob")

