import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob
import re

# Set page config
st.set_page_config(
    page_title="V1 Pharma Call Center Quality Control",
    page_icon="📊",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    clinical_df = pd.read_csv('V1 Pharma Clinical.csv')
    admin_df = pd.read_csv('V1 Pharma Admin.csv')

    clinical_df.columns = clinical_df.columns.str.strip()
    admin_df.columns = admin_df.columns.str.strip()

    clinical_df['Date of the call'] = pd.to_datetime(clinical_df['Date of the call'], errors='coerce')
    clinical_df['Date of the evaluation'] = pd.to_datetime(clinical_df['Date of the evaluation'], errors='coerce')
    admin_df['Date of the call'] = pd.to_datetime(admin_df['Date of the call'], errors='coerce')
    admin_df['Date of the evaluation'] = pd.to_datetime(admin_df['Date of the evaluation'], errors='coerce')

    clinical_df = clinical_df.dropna(subset=['Agent name'])
    clinical_df = clinical_df[clinical_df['Agent name'] != '']
    admin_df = admin_df.dropna(subset=['Agent name'])
    admin_df = admin_df[admin_df['Agent name'] != '']

    clinical_df['Dataset'] = 'Clinical'
    admin_df['Dataset'] = 'Admin'

    combined_df = pd.concat([clinical_df, admin_df], ignore_index=True)

    return clinical_df, admin_df, combined_df

# Sentiment analysis
def analyze_sentiment(text):
    if pd.isna(text) or text == '':
        return 'Neutral', 0.0
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Performance categorization
def categorize_performance(score):
    if score >= 95:
        return 'Excellent'
    elif score >= 85:
        return 'Good'
    elif score >= 75:
        return 'Average'
    else:
        return 'Needs Improvement'

# Training issues extraction
def extract_training_issues(text):
    if pd.isna(text):
        return []
    issues = []
    text_lower = text.lower()

    if any(word in text_lower for word in ['training', 'need', 'training issues', 'behavioral issues']):
        issues.append('Training Required')

    if any(word in text_lower for word in ['verification', 'verify', 'id number', 'dob', 'member verification']):
        issues.append('Verification Issues')

    if any(word in text_lower for word in ['documentation', 'notes', 'dmp', 'call note', 'record']):
        issues.append('Documentation Issues')

    if any(word in text_lower for word in ['communication', 'talking over', 'listening', 'active listening']):
        issues.append('Communication Issues')

    if any(word in text_lower for word in ['empathy', 'understanding', 'reassure', 'support']):
        issues.append('Empathy Issues')

    if any(word in text_lower for word in ['compliance', 'sop', 'procedure', 'protocol']):
        issues.append('Compliance Issues')

    if any(word in text_lower for word in ['disclaimer', 'mention disclaimer']):
        issues.append('Disclaimer Issues')

    if any(word in text_lower for word in ['professional', 'tone', 'etiquette', 'behavior']):
        issues.append('Professionalism Issues')

    if any(word in text_lower for word in ['hold procedure', 'hold']):
        issues.append('Hold Procedure Issues')

    if any(word in text_lower for word in ['reference number', 'call reference']):
        issues.append('Reference Number Issues')

    return issues

# Process data
@st.cache_data
def process_data(df):
    df['Summary Sentiment'], df['Sentiment Score'] = zip(*df['Summary of call'].apply(analyze_sentiment))
    if 'AI Analysis' in df.columns:
        df['AI Analysis Sentiment'], df['AI Sentiment Score'] = zip(*df['AI Analysis'].apply(analyze_sentiment))

    df['Performance Category'] = df['Grand Total'].apply(categorize_performance)

    if 'AI Analysis' in df.columns:
        df['Training Issues'] = df['AI Analysis'].apply(extract_training_issues)
    else:
        df['Training Issues'] = [[] for _ in range(len(df))]

    return df

# Load and process data
clinical_df, admin_df, combined_df = load_data()
clinical_processed = process_data(clinical_df)
admin_processed = process_data(admin_df)
combined_processed = process_data(combined_df)

# Main header
st.markdown('<h1 class="main-header">📊 V1 Pharma Call Center Quality Control</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ Controls")

dataset_option = st.sidebar.selectbox(
    "Dataset",
    ["Combined", "Clinical", "Admin"]
)

if dataset_option == "Clinical":
    df = clinical_processed
elif dataset_option == "Admin":
    df = admin_processed
else:
    df = combined_processed

# Filters
if not df['Date of the call'].isna().all():
    min_date = df['Date of the call'].min().date()
    max_date = df['Date of the call'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        df = df[(df['Date of the call'].dt.date >= date_range[0]) & (df['Date of the call'].dt.date <= date_range[1])]

agents = sorted(df['Agent name'].unique())
selected_agents = st.sidebar.multiselect(
    "Agents",
    agents,
    default=agents
)
if selected_agents:
    df = df[df['Agent name'].isin(selected_agents)]

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Calls", len(df))

with col2:
    avg_score = df['Grand Total'].mean()
    st.metric("Average Score", f"{avg_score:.1f}/100")

with col3:
    st.metric("Active Agents", df['Agent name'].nunique())

with col4:
    excellent_pct = (df['Performance Category'] == 'Excellent').mean() * 100
    st.metric("Excellent Rate", f"{excellent_pct:.1f}%")

# Performance Distribution
st.header("📈 Performance Overview")

perf_counts = df['Performance Category'].value_counts()
fig_perf = px.pie(
    values=perf_counts.values,
    names=perf_counts.index,
    title="Performance Distribution",
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_perf, use_container_width=True)

# Agent Performance
agent_stats = df.groupby('Agent name')['Grand Total'].agg(['mean', 'count']).sort_values('mean', ascending=True)
fig_agents = px.bar(
    agent_stats,
    x='mean',
    y=agent_stats.index,
    orientation='h',
    title="Agent Performance Ranking",
    labels={'mean': 'Average Score'},
    color='mean',
    color_continuous_scale='RdYlGn'
)
fig_agents.add_vline(x=df['Grand Total'].mean(), line_dash="dash", line_color="red")
st.plotly_chart(fig_agents, use_container_width=True)

# Sentiment Analysis
st.header("💭 Sentiment Analysis")

summary_sentiment = df['Summary Sentiment'].value_counts()
fig_sentiment = px.bar(
    summary_sentiment,
    title="Call Summary Sentiment",
    labels={'value': 'Calls', 'index': 'Sentiment'},
    color=summary_sentiment.index,
    color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# Training Issues
st.header("🎓 Training Issues")

if df['Training Issues'].apply(len).sum() > 0:
    all_issues = [issue for issues in df['Training Issues'] for issue in issues]
    if all_issues:
        issues_series = pd.Series(all_issues).value_counts()

        fig_issues = px.bar(
            issues_series,
            title="Training Issues Identified",
            labels={'value': 'Frequency'},
            color=issues_series.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_issues, use_container_width=True)

        st.subheader("Priority Recommendations")
        recommendations = {
            'Verification Issues': 'Implement verification checklists and DOB/ID confirmation training.',
            'Communication Issues': 'Focus on active listening and clear information delivery.',
            'Documentation Issues': 'Standardize DMP notes and call logging procedures.',
            'Disclaimer Issues': 'Create automated disclaimer reminders.',
            'Professionalism Issues': 'Develop tone and etiquette training.',
            'Empathy Issues': 'Implement customer-centric communication training.',
            'Compliance Issues': 'Establish SOP compliance training programs.',
            'Hold Procedure Issues': 'Train on proper hold protocols.',
            'Reference Number Issues': 'Implement automated reference number systems.',
            'Training Required': 'General skills enhancement needed.'
        }

        for issue in issues_series.head(3).index:
            if issue in recommendations:
                st.markdown(f"**{issue}**: {recommendations[issue]}")

# Detailed Analysis
st.header("📋 Detailed Call Analysis")

display_df = df[['Agent name', 'Date of the call', 'Query type', 'Grand Total', 'Performance Category', 'Summary Sentiment']].head(50)
st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**V1 Pharma Quality Control System** - Data-driven insights for call center excellence")