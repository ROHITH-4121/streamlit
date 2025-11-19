import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import requests
from urllib.parse import urlparse
import json

# Page configuration
st.set_page_config(
    page_title="TruthShield - AI Misinformation Detector",
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
        text-align: center;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-danger {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    .alert-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    .alert-success {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        color: #065f46;
    }
    </style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.suspicious_patterns = [
            r'\b(shocking|unbelievable|you won\'t believe|miraculous|secret)\b',
            r'\b(doctors hate|they don\'t want you to know|banned|censored)\b',
            r'\b(breaking|urgent|alert|warning)\b.*!+',
            r'!!!+',
            r'\bALL CAPS\b.*[A-Z]{10,}',
        ]
        
        self.clickbait_keywords = [
            'shocking', 'unbelievable', 'amazing', 'incredible', 'you won\'t believe',
            'what happens next', 'jaw-dropping', 'mind-blowing', 'stunning', 'miraculous'
        ]
        
        self.emotional_manipulation = [
            'outrage', 'furious', 'destroyed', 'slammed', 'devastated',
            'terrifying', 'horrific', 'nightmare', 'disaster', 'catastrophe'
        ]
        
        self.unreliable_domains = [
            'fake-news.com', 'conspiracy-central.net', 'truth-exposed.org',
            'real-news-not-fake.com', 'patriot-eagle-news.com'
        ]

    def analyze_text_features(self, text):
        """Analyze linguistic features of the text"""
        features = {}
        
        # Basic metrics
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Pattern detection
        features['suspicious_patterns'] = sum(1 for pattern in self.suspicious_patterns 
                                             if re.search(pattern, text, re.IGNORECASE))
        
        # Clickbait detection
        text_lower = text.lower()
        features['clickbait_score'] = sum(1 for keyword in self.clickbait_keywords 
                                         if keyword in text_lower)
        
        # Emotional manipulation
        features['emotion_score'] = sum(1 for keyword in self.emotional_manipulation 
                                       if keyword in text_lower)
        
        return features

    def detect_bias_indicators(self, text):
        """Detect potential bias indicators"""
        bias_indicators = {
            'loaded_language': 0,
            'one_sided': 0,
            'lacks_sources': 0,
            'emotional_appeal': 0
        }
        
        # Loaded language patterns
        loaded_terms = ['allegedly', 'supposedly', 'claims', 'reportedly', 'rumored']
        bias_indicators['loaded_language'] = sum(1 for term in loaded_terms 
                                                 if term in text.lower())
        
        # Check for lack of attribution
        if not re.search(r'(according to|says|stated|reported|source)', text, re.IGNORECASE):
            bias_indicators['lacks_sources'] = 1
        
        # Emotional appeal
        emotion_words = self.emotional_manipulation
        bias_indicators['emotional_appeal'] = sum(1 for word in emotion_words 
                                                  if word in text.lower())
        
        return bias_indicators

    def analyze_url_credibility(self, url):
        """Analyze URL for credibility signals"""
        if not url:
            return {'credibility_score': 50, 'warnings': ['No URL provided']}
        
        credibility_score = 100
        warnings = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check against known unreliable sources
            if any(unreliable in domain for unreliable in self.unreliable_domains):
                credibility_score -= 40
                warnings.append('Domain flagged as potentially unreliable')
            
            # Check for suspicious TLD
            suspicious_tlds = ['.xyz', '.top', '.click', '.link']
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                credibility_score -= 15
                warnings.append('Suspicious top-level domain')
            
            # Check for HTTPS
            if parsed.scheme != 'https':
                credibility_score -= 10
                warnings.append('No HTTPS encryption')
            
            # Check for excessive subdomains
            if domain.count('.') > 2:
                credibility_score -= 10
                warnings.append('Unusual subdomain structure')
                
        except Exception as e:
            warnings.append(f'Error parsing URL: {str(e)}')
            credibility_score = 50
        
        return {
            'credibility_score': max(0, credibility_score),
            'warnings': warnings
        }

    def extract_claims(self, text):
        """Extract potential factual claims from text"""
        claims = []
        
        # Simple claim extraction based on patterns
        claim_patterns = [
            r'(?:study|research|report) (?:shows|finds|reveals|proves) that ([^.!?]+)',
            r'(?:scientists|experts|doctors) (?:say|claim|discovered) that ([^.!?]+)',
            r'([0-9]+%?) (?:of|percent) ([^.!?]+)',
            r'it is (?:a fact|proven|confirmed) that ([^.!?]+)'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(match.group(0))
        
        return claims[:5]  # Return top 5 claims

    def calculate_credibility_score(self, features, bias_indicators, url_analysis):
        """Calculate overall credibility score"""
        score = 100
        
        # Deduct based on features
        score -= features['exclamation_count'] * 2
        score -= features['suspicious_patterns'] * 15
        score -= features['clickbait_score'] * 10
        score -= features['emotion_score'] * 8
        score -= features['caps_ratio'] * 50
        
        # Deduct based on bias
        score -= sum(bias_indicators.values()) * 5
        
        # Factor in URL credibility
        score = (score + url_analysis['credibility_score']) / 2
        
        return max(0, min(100, score))

    def get_credibility_level(self, score):
        """Convert score to credibility level"""
        if score >= 80:
            return "HIGH CREDIBILITY", "success"
        elif score >= 60:
            return "MODERATE CREDIBILITY", "warning"
        elif score >= 40:
            return "LOW CREDIBILITY", "warning"
        else:
            return "VERY LOW CREDIBILITY", "danger"

# Initialize detector
detector = FakeNewsDetector()

# Header
st.markdown('<h1 class="main-header">🛡️ TruthShield</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Fake News & Misinformation Detector</p>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/security-shield-green.png", width=150)
    st.title("Analysis Options")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Text Analysis", "URL Analysis", "Batch Analysis"]
    )
    
    st.markdown("---")
    st.markdown("### About TruthShield")
    st.info("""
    TruthShield uses advanced AI algorithms to analyze content for:
    - Clickbait detection
    - Emotional manipulation
    - Source credibility
    - Bias indicators
    - Factual claim extraction
    """)
    
    st.markdown("---")
    st.markdown("### Detection Metrics")
    st.markdown("""
    - **Linguistic Analysis**: Pattern recognition
    - **Sentiment Analysis**: Emotional manipulation detection
    - **Source Verification**: Domain credibility check
    - **Bias Detection**: One-sided reporting indicators
    """)

# Main content area
if analysis_type == "Text Analysis":
    st.header("📝 Text Content Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Paste the article or news content here:",
            height=250,
            placeholder="Enter the text you want to analyze..."
        )
    
    with col2:
        url_input = st.text_input(
            "Source URL (optional):",
            placeholder="https://example.com/article"
        )
        
        st.markdown("### Quick Stats")
        if text_input:
            st.metric("Words", len(text_input.split()))
            st.metric("Characters", len(text_input))
    
    if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
        if text_input:
            with st.spinner("Analyzing content with AI algorithms..."):
                # Perform analysis
                features = detector.analyze_text_features(text_input)
                bias_indicators = detector.detect_bias_indicators(text_input)
                url_analysis = detector.analyze_url_credibility(url_input)
                claims = detector.extract_claims(text_input)
                credibility_score = detector.calculate_credibility_score(
                    features, bias_indicators, url_analysis
                )
                credibility_level, alert_type = detector.get_credibility_level(credibility_score)
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Credibility Score Card
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h2>{credibility_score:.1f}/100</h2>
                            <p>Credibility Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h2>{credibility_level.split()[0]}</h2>
                            <p>Trust Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    risk_level = "Low" if credibility_score > 70 else "Medium" if credibility_score > 40 else "High"
                    st.markdown(f"""
                        <div class="metric-card">
                            <h2>{risk_level}</h2>
                            <p>Risk Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Alert Box
                st.markdown(f"""
                    <div class="alert-box alert-{alert_type}">
                        <strong>⚠️ {credibility_level}</strong><br>
                        {'This content shows signs of reliability and factual reporting.' if credibility_score > 70 else
                         'This content shows some concerning patterns. Verify with additional sources.' if credibility_score > 40 else
                         'This content shows multiple red flags. Exercise extreme caution.'}
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Linguistic Analysis", 
                    "🎯 Bias Detection", 
                    "🔗 Source Credibility",
                    "💡 Extracted Claims"
                ])
                
                with tab1:
                    st.subheader("Linguistic Features")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create metrics dataframe
                        metrics_data = {
                            'Metric': ['Word Count', 'Exclamation Marks', 'Suspicious Patterns', 
                                      'Clickbait Indicators', 'Emotional Language'],
                            'Value': [features['word_count'], features['exclamation_count'],
                                     features['suspicious_patterns'], features['clickbait_score'],
                                     features['emotion_score']],
                            'Risk': ['Low' if features['exclamation_count'] < 3 else 'High',
                                    'Low' if features['exclamation_count'] < 3 else 'High',
                                    'Low' if features['suspicious_patterns'] == 0 else 'High',
                                    'Low' if features['clickbait_score'] == 0 else 'High',
                                    'Low' if features['emotion_score'] < 2 else 'High']
                        }
                        st.dataframe(metrics_data, use_container_width=True)
                    
                    with col2:
                        # Gauge chart for caps ratio
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=features['caps_ratio'] * 100,
                            title={'text': "CAPS Usage %"},
                            delta={'reference': 10},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgreen"},
                                    {'range': [10, 30], 'color': "yellow"},
                                    {'range': [30, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 30
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Bias Indicators")
                    
                    bias_df = pd.DataFrame({
                        'Indicator': ['Loaded Language', 'One-Sided Reporting', 
                                     'Lacks Sources', 'Emotional Appeal'],
                        'Count': list(bias_indicators.values())
                    })
                    
                    fig = px.bar(bias_df, x='Indicator', y='Count', 
                                color='Count',
                                color_continuous_scale=['green', 'yellow', 'red'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    total_bias = sum(bias_indicators.values())
                    if total_bias > 3:
                        st.warning("⚠️ High bias indicators detected. Content may lack objectivity.")
                    elif total_bias > 1:
                        st.info("ℹ️ Some bias indicators present. Cross-reference with other sources.")
                    else:
                        st.success("✅ Minimal bias indicators detected.")
                
                with tab3:
                    st.subheader("Source Credibility Analysis")
                    
                    if url_input:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("URL Credibility Score", 
                                     f"{url_analysis['credibility_score']}/100")
                            
                            parsed_url = urlparse(url_input)
                            st.write(f"**Domain:** {parsed_url.netloc}")
                            st.write(f"**Protocol:** {parsed_url.scheme}")
                        
                        with col2:
                            if url_analysis['warnings']:
                                st.warning("**Warnings:**")
                                for warning in url_analysis['warnings']:
                                    st.write(f"• {warning}")
                            else:
                                st.success("✅ No major credibility issues detected")
                    else:
                        st.info("No URL provided for analysis")
                
                with tab4:
                    st.subheader("Extracted Factual Claims")
                    
                    if claims:
                        st.write("The following claims were identified and should be fact-checked:")
                        for i, claim in enumerate(claims, 1):
                            st.markdown(f"""
                                <div style="background-color: #f0f9ff; padding: 10px; 
                                     border-radius: 5px; margin: 10px 0; border-left: 3px solid #3b82f6;">
                                    <strong>Claim {i}:</strong> {claim}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.info("💡 **Tip:** Verify these claims using fact-checking websites like Snopes, FactCheck.org, or PolitiFact")
                    else:
                        st.info("No specific factual claims detected in the text")
                
                # Recommendations
                st.markdown("---")
                st.header("📋 Recommendations")
                
                if credibility_score > 70:
                    st.success("""
                    ✅ **This content appears relatively credible**, but always:
                    - Verify with multiple sources
                    - Check the publication date
                    - Look for author credentials
                    """)
                elif credibility_score > 40:
                    st.warning("""
                    ⚠️ **Exercise caution with this content:**
                    - Cross-reference with established news sources
                    - Look for original sources and citations
                    - Be aware of potential bias
                    - Verify any statistics or claims
                    """)
                else:
                    st.error("""
                    🚨 **High risk of misinformation:**
                    - Do not share without verification
                    - Check fact-checking websites
                    - Look for coverage by reputable news organizations
                    - Be extremely skeptical of claims made
                    """)
        else:
            st.warning("Please enter some text to analyze")

elif analysis_type == "URL Analysis":
    st.header("🔗 URL & Domain Analysis")
    
    url_input = st.text_input(
        "Enter article URL:",
        placeholder="https://example.com/article/123"
    )
    
    if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
        if url_input:
            with st.spinner("Analyzing URL and domain credibility..."):
                url_analysis = detector.analyze_url_credibility(url_input)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                parsed_url = urlparse(url_input)
                
                with col1:
                    st.metric("Domain Credibility", f"{url_analysis['credibility_score']}/100")
                
                with col2:
                    st.metric("Protocol", parsed_url.scheme.upper())
                
                with col3:
                    tld = parsed_url.netloc.split('.')[-1]
                    st.metric("TLD", f".{tld}")
                
                st.markdown("---")
                
                st.subheader("Domain Information")
                st.write(f"**Full Domain:** {parsed_url.netloc}")
                st.write(f"**Path:** {parsed_url.path or '/'}")
                
                if url_analysis['warnings']:
                    st.subheader("⚠️ Warnings")
                    for warning in url_analysis['warnings']:
                        st.warning(warning)
                else:
                    st.success("✅ No major issues detected with this URL")
        else:
            st.warning("Please enter a URL to analyze")

else:  # Batch Analysis
    st.header("📦 Batch Analysis")
    st.info("Analyze multiple articles at once")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with columns: 'text' and 'url' (optional)",
        type=['csv']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} articles")
        
        if st.button("🔍 Analyze All", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                text = row.get('text', '')
                url = row.get('url', '')
                
                if text:
                    features = detector.analyze_text_features(text)
                    bias_indicators = detector.detect_bias_indicators(text)
                    url_analysis = detector.analyze_url_credibility(url)
                    score = detector.calculate_credibility_score(
                        features, bias_indicators, url_analysis
                    )
                    
                    results.append({
                        'Article': idx + 1,
                        'Credibility Score': score,
                        'Word Count': features['word_count'],
                        'Red Flags': features['suspicious_patterns'] + features['clickbait_score']
                    })
                
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            
            st.subheader("📊 Batch Analysis Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Credibility", f"{results_df['Credibility Score'].mean():.1f}")
            
            with col2:
                high_risk = len(results_df[results_df['Credibility Score'] < 40])
                st.metric("High Risk Articles", high_risk)
            
            with col3:
                st.metric("Total Analyzed", len(results_df))
            
            # Distribution chart
            fig = px.histogram(results_df, x='Credibility Score', 
                              title='Credibility Score Distribution',
                              nbins=20)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a CSV file to perform batch analysis")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 20px;">
        <p><strong>TruthShield v2.0</strong> | AI-Powered Misinformation Detection</p>
        <p style="font-size: 0.9em;">⚠️ This tool provides automated analysis. Always verify important information with multiple credible sources.</p>
    </div>
""", unsafe_allow_html=True)
