# TruthShield: AI-Powered Fake News Detector - Streamlit App
# Installation: pip install streamlit pandas numpy scikit-learn nltk plotly wordcloud pillow

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pickle
from datetime import datetime
import io
from PIL import Image

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="TruthShield - Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fake-news {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .real-news {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

class TruthShieldDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [self.ps.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def train(self, df, selected_model='Logistic Regression'):
        """Train the selected model"""
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train selected model
        model = self.models[selected_model]
        model.fit(X_train, y_train)
        
        self.best_model = model
        self.best_model_name = selected_model
        self.is_trained = True
        
        # Get metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'df': df
        }
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.is_trained:
            return None
            
        cleaned = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.best_model.predict(vectorized)[0]
        probability = self.best_model.predict_proba(vectorized)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'label': prediction,
            'confidence': max(probability) * 100,
            'fake_probability': probability[1] * 100,
            'real_probability': probability[0] * 100
        }

@st.cache_data
def create_sample_dataset():
    """Create sample dataset for demonstration"""
    data = {
        'text': [
            # Fake news examples
            "Breaking: Scientists discover that drinking bleach cures all diseases instantly",
            "Shocking revelation: Moon landing was completely staged in Hollywood studios",
            "Exclusive: Celebrities caught in massive scandal that will shock you",
            "Miracle cure found: This one weird trick doctors don't want you to know",
            "Urgent: Government hiding alien contact for 50 years, insider reveals",
            "Unbelievable: New study shows vaccines contain mind control chips",
            "Alert: 5G towers are causing coronavirus spread worldwide",
            "Breaking news: President secretly replaced by clone, sources claim",
            "You won't believe what happened next! Doctors hate this simple trick",
            "Conspiracy revealed: Water is making frogs change their behavior patterns",
            
            # Real news examples
            "Stock market sees modest gains amid economic recovery efforts",
            "New climate report shows rising global temperatures over past decade",
            "Local community center opens new programs for senior citizens",
            "Research team publishes peer-reviewed study on cancer treatment advances",
            "City council votes to approve new infrastructure budget plan",
            "Technology company announces quarterly earnings meeting analyst expectations",
            "University study reveals correlation between exercise and mental health",
            "International summit concludes with agreement on trade regulations",
            "Federal Reserve announces interest rate decision after policy meeting",
            "New archaeological discovery sheds light on ancient civilization practices"
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 1 = Fake
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = Real
    }
    return pd.DataFrame(data)

def create_wordcloud(text, colormap='Reds'):
    """Generate word cloud"""
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=colormap).generate(text)
    return wordcloud

def plot_confusion_matrix(cm):
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Real', 'Fake'],
        y=['Real', 'Fake'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        font=dict(size=14)
    )
    return fig

def plot_confidence_distribution(df):
    """Plot confidence score distribution"""
    fig = px.histogram(df, x='confidence', color='label_text',
                      nbins=20,
                      title='Confidence Score Distribution',
                      labels={'confidence': 'Confidence Score (%)', 'count': 'Number of Articles'},
                      color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'})
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🛡️ TruthShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fake News & Misinformation Detector</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = TruthShieldDetector()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=100)
        st.title("Control Panel")
        
        page = st.radio("Navigation", 
                       ["🏠 Home", "🤖 Train Model", "🔍 Detect News", "📊 Analytics", "📜 History"],
                       label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        TruthShield uses machine learning to detect fake news and misinformation.
        
        **Features:**
        - Multiple ML algorithms
        - Real-time detection
        - Confidence scoring
        - Detailed analytics
        """)
        
        st.markdown("---")
        st.markdown("### Model Status")
        if st.session_state.trained:
            st.success("✅ Model Trained")
            if st.session_state.detector.best_model_name:
                st.write(f"**Algorithm:** {st.session_state.detector.best_model_name}")
            if st.session_state.results:
                st.write(f"**Accuracy:** {st.session_state.results['accuracy']*100:.2f}%")
        else:
            st.warning("⚠️ Model Not Trained")
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home()
    elif page == "🤖 Train Model":
        show_train_model()
    elif page == "🔍 Detect News":
        show_detect_news()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "📜 History":
        show_history()

def show_home():
    """Home page"""
    st.markdown("## Welcome to TruthShield")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What is TruthShield?")
        st.write("""
        TruthShield is an advanced AI-powered tool designed to combat misinformation 
        by detecting fake news articles with high accuracy. Using machine learning 
        algorithms and natural language processing, it analyzes text content to 
        identify patterns commonly associated with misleading information.
        """)
        
        st.markdown("### 🚀 How to Get Started")
        st.write("""
        1. **Train Model**: Upload your dataset or use sample data
        2. **Detect News**: Paste any news article to check authenticity
        3. **View Analytics**: Explore detailed insights and visualizations
        4. **Check History**: Review past predictions
        """)
    
    with col2:
        st.markdown("### 📈 Key Features")
        st.write("""
        - **Multiple Algorithms**: Choose from Logistic Regression, Naive Bayes, or Random Forest
        - **Real-time Detection**: Instant analysis of news articles
        - **Confidence Scores**: Understand how certain the model is
        - **Interactive Visualizations**: Explore data with beautiful charts
        - **Prediction History**: Track all your analyses
        """)
        
        st.markdown("### 🔒 Why Trust Matters")
        st.write("""
        In today's digital age, misinformation spreads faster than ever. 
        TruthShield empowers you to:
        - Make informed decisions
        - Verify sources before sharing
        - Combat the spread of fake news
        - Protect yourself and others from manipulation
        """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start by training the model with sample data in the 'Train Model' section!")

def show_train_model():
    """Train model page"""
    st.markdown("## 🤖 Train Detection Model")
    
    tab1, tab2 = st.tabs(["📁 Use Sample Data", "📤 Upload Your Data"])
    
    with tab1:
        st.markdown("### Sample Dataset")
        st.write("Train the model using our pre-loaded sample dataset containing 20 articles.")
        
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Select ML Algorithm",
                ["Logistic Regression", "Naive Bayes", "Random Forest"],
                help="Choose which machine learning algorithm to use for training"
            )
        
        if st.button("🚀 Train Model with Sample Data", type="primary"):
            with st.spinner("Training model... Please wait"):
                df = create_sample_dataset()
                
                # Display dataset info
                st.success(f"✅ Loaded {len(df)} articles")
                st.dataframe(df.head(), use_container_width=True)
                
                # Train model
                results = st.session_state.detector.train(df, model_choice)
                st.session_state.results = results
                st.session_state.trained = True
                
                # Show results
                st.balloons()
                st.success(f"🎉 Model trained successfully with {model_choice}!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Training Samples", len(results['df']))
                with col3:
                    fake_count = len(results['df'][results['df']['label'] == 1])
                    st.metric("Fake News Samples", fake_count)
                
                # Confusion matrix
                st.plotly_chart(plot_confusion_matrix(results['confusion_matrix']), 
                              use_container_width=True)
    
    with tab2:
        st.markdown("### Upload Your Dataset")
        st.write("Upload a CSV file with columns: 'text' (article content) and 'label' (0=Real, 1=Fake)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                if 'text' not in df.columns or 'label' not in df.columns:
                    st.error("❌ CSV must contain 'text' and 'label' columns!")
                else:
                    st.success(f"✅ Loaded {len(df)} articles")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    model_choice = st.selectbox(
                        "Select ML Algorithm",
                        ["Logistic Regression", "Naive Bayes", "Random Forest"],
                        key="upload_model"
                    )
                    
                    if st.button("🚀 Train Model with Uploaded Data", type="primary"):
                        with st.spinner("Training model..."):
                            results = st.session_state.detector.train(df, model_choice)
                            st.session_state.results = results
                            st.session_state.trained = True
                            
                            st.balloons()
                            st.success(f"🎉 Model trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                            with col2:
                                st.metric("Training Samples", len(results['df']))
                            with col3:
                                fake_count = len(results['df'][results['df']['label'] == 1])
                                st.metric("Fake News Samples", fake_count)
                            
                            st.plotly_chart(plot_confusion_matrix(results['confusion_matrix']), 
                                          use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

def show_detect_news():
    """Detect news page"""
    st.markdown("## 🔍 Detect Fake News")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section!")
        return
    
    st.write("Paste any news article below to check if it's fake or real.")
    
    # Text input
    article_text = st.text_area(
        "Enter News Article",
        height=200,
        placeholder="Paste the news article text here...",
        help="Enter the complete text of the news article you want to verify"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    if analyze_button:
        if not article_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("Analyzing article..."):
                result = st.session_state.detector.predict(article_text)
                
                if result:
                    # Save to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'text': article_text[:100] + "...",
                        'prediction': result['prediction'],
                        'confidence': result['confidence']
                    })
                    
                    # Display result
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    if result['prediction'] == 'FAKE':
                        st.markdown(f"""
                        <div class="fake-news">
                            <h2 style="color: #F44336;">⚠️ FAKE NEWS DETECTED</h2>
                            <p style="font-size: 1.2rem;">This article shows characteristics of misinformation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="real-news">
                            <h2 style="color: #4CAF50;">✅ LIKELY REAL NEWS</h2>
                            <p style="font-size: 1.2rem;">This article appears to be authentic.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Confidence", f"{result['confidence']:.1f}%")
                    with col2:
                        st.metric("Fake Probability", f"{result['fake_probability']:.1f}%")
                    with col3:
                        st.metric("Real Probability", f"{result['real_probability']:.1f}%")
                    
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#F44336" if result['prediction'] == 'FAKE' else "#4CAF50"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendation")
                    if result['confidence'] > 80:
                        st.info(f"The model is highly confident ({result['confidence']:.1f}%) about this classification.")
                    elif result['confidence'] > 60:
                        st.warning(f"The model is moderately confident ({result['confidence']:.1f}%). Consider cross-checking with other sources.")
                    else:
                        st.warning(f"The model has low confidence ({result['confidence']:.1f}%). This article requires additional verification from multiple sources.")
    
    # Quick test examples
    with st.expander("📝 Try Sample Articles"):
        st.markdown("#### Sample Fake News")
        fake_sample = "Breaking: Scientists discover that drinking bleach cures all diseases instantly. Doctors don't want you to know this one weird trick!"
        if st.button("Test Fake News Sample"):
            st.session_state.sample_text = fake_sample
            st.rerun()
        
        st.markdown("#### Sample Real News")
        real_sample = "The Federal Reserve announced today that interest rates will remain unchanged following their monthly policy meeting. Economic analysts had predicted this decision based on recent inflation data."
        if st.button("Test Real News Sample"):
            st.session_state.sample_text = real_sample
            st.rerun()

def show_analytics():
    """Analytics page"""
    st.markdown("## 📊 Model Analytics")
    
    if not st.session_state.trained or not st.session_state.results:
        st.warning("⚠️ Please train the model first to view analytics!")
        return
    
    results = st.session_state.results
    df = results['df']
    
    # Add predictions to dataframe
    df['label_text'] = df['label'].map({0: 'Real', 1: 'Fake'})
    
    predictions = []
    confidences = []
    for text in df['text']:
        pred = st.session_state.detector.predict(text)
        predictions.append(pred['prediction'])
        confidences.append(pred['confidence'])
    
    df['predicted'] = predictions
    df['confidence'] = confidences
    
    # Overview metrics
    st.markdown("### 📈 Model Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
    with col2:
        st.metric("Total Articles", len(df))
    with col3:
        fake_count = len(df[df['label'] == 1])
        st.metric("Fake News", fake_count)
    with col4:
        real_count = len(df[df['label'] == 0])
        st.metric("Real News", real_count)
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "☁️ Word Clouds", "📉 Data Distribution"])
    
    with tab1:
        st.markdown("### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(results['confusion_matrix']), 
                       use_container_width=True)
        
        st.markdown("### Confidence Score Distribution")
        st.plotly_chart(plot_confidence_distribution(df), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fake News Word Cloud")
            fake_text = ' '.join(df[df['label'] == 1]['cleaned_text'])
            if fake_text.strip():
                wordcloud_fake = create_wordcloud(fake_text, 'Reds')
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_fake, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.markdown("### Real News Word Cloud")
            real_text = ' '.join(df[df['label'] == 0]['cleaned_text'])
            if real_text.strip():
                wordcloud_real = create_wordcloud(real_text, 'Greens')
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_real, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    
    with tab3:
        st.markdown("### Label Distribution")
        fig = px.pie(df, names='label_text', title='Fake vs Real News Distribution',
                    color='label_text',
                    color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Article Statistics")
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['text'].apply(len)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Word Count by Type', 'Character Count by Type'))
        
        for label in [0, 1]:
            label_text = 'Real' if label == 0 else 'Fake'
            color = '#4CAF50' if label == 0 else '#F44336'
            data = df[df['label'] == label]['word_count']
            fig.add_trace(go.Box(y=data, name=label_text, marker_color=color), row=1, col=1)
            
            data = df[df['label'] == label]['char_count']
            fig.add_trace(go.Box(y=data, name=label_text, marker_color=color, showlegend=False), row=1, col=2)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"truthshield_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary = f"""
TruthShield Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model: {st.session_state.detector.best_model_name}
Accuracy: {results['accuracy']*100:.2f}%
Total Articles: {len(df)}
Fake News: {fake_count}
Real News: {real_count}
        """
        st.download_button(
            label="📄 Download Report",
            data=summary,
            file_name=f"truthshield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def show_history():
    """History page"""
    st.markdown("## 📜 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Start analyzing articles in the 'Detect News' section!")
        return
    
    st.write(f"Total predictions: {len(st.session_state.prediction_history)}")
    
    # Display history
    for i, pred in enumerate(reversed(st.session_state.prediction_history)):
        with st.expander(f"#{len(st.session_state.prediction_history) - i} - {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Article Preview:**")
                st.write(pred['text'])
            
            with col2:
                if pred['prediction'] == 'FAKE':
                    st.error(f"**{pred['prediction']}**")
                else:
                    st.success(f"**{pred['prediction']}**")
                st.metric("Confidence", f"{pred['confidence']:.1f}%")
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.rerun()

# Import matplotlib for word clouds
import matplotlib.pyplot as plt

if __name__ == "__main__":
    main()