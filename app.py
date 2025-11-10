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

# Business Intelligence Section
st.header("💼 Business Intelligence & Decision Support")

# Cost-Benefit Analysis
st.subheader("💰 Quality Improvement ROI Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Training Investment Calculator
    st.markdown("### Training Investment")
    avg_calls_per_agent = len(df) / df['Agent name'].nunique()
    training_cost_per_agent = st.number_input("Training Cost per Agent (R)", value=5000, min_value=0)
    total_training_investment = training_cost_per_agent * df['Agent name'].nunique()
    st.metric("Total Training Investment", f"R{total_training_investment:,.0f}")

with col2:
    # Quality Improvement Projection
    current_avg_score = df['Grand Total'].mean()
    projected_improvement = st.slider("Expected Score Improvement (%)", 5, 25, 15)
    projected_avg_score = current_avg_score * (1 + projected_improvement/100)
    st.metric("Projected Average Score", f"{projected_avg_score:.1f}/100", f"+{projected_improvement}%")

with col3:
    # ROI Calculation
    calls_per_month = len(df) * 2  # Assuming 2-month data period
    cost_per_call = st.number_input("Cost per Call (R)", value=50, min_value=0)
    monthly_call_cost = calls_per_month * cost_per_call
    quality_improvement_value = monthly_call_cost * (projected_improvement/100)
    roi = (quality_improvement_value / total_training_investment) * 100 if total_training_investment > 0 else 0
    st.metric("Projected ROI", f"{roi:.1f}%", "Quality Improvement Value")

# Workforce Optimization
st.subheader("👥 Workforce Optimization Insights")

col1, col2 = st.columns(2)

with col1:
    # Agent Utilization Analysis
    agent_workload = df.groupby('Agent name').size()
    avg_workload = agent_workload.mean()
    workload_std = agent_workload.std()

    fig_workload = px.bar(
        agent_workload.sort_values(ascending=True),
        title="Agent Call Volume Distribution",
        labels={'value': 'Number of Calls', 'Agent name': 'Agent'},
        color=agent_workload.values,
        color_continuous_scale='Blues'
    )
    fig_workload.add_hline(y=avg_workload, line_dash="dash", line_color="red",
                          annotation_text=f"Average: {avg_workload:.1f}")
    st.plotly_chart(fig_workload, use_container_width=True)

with col2:
    # Performance vs Volume Analysis
    agent_perf_volume = df.groupby('Agent name').agg({
        'Grand Total': 'mean',
        'Agent name': 'size'
    }).rename(columns={'Agent name': 'Call Volume'})

    fig_perf_volume = px.scatter(
        agent_perf_volume,
        x='Call Volume',
        y='Grand Total',
        title="Performance vs Call Volume",
        labels={'Grand Total': 'Average Score', 'Call Volume': 'Number of Calls'},
        color='Grand Total',
        color_continuous_scale='RdYlGn',
        size='Call Volume'
    )
    st.plotly_chart(fig_perf_volume, use_container_width=True)

# Predictive Analytics
st.subheader("🔮 Predictive Quality Insights")

# Risk Assessment
risk_factors = []

# High volume low performance agents
high_volume_threshold = agent_workload.quantile(0.75)
low_perf_threshold = df['Grand Total'].mean() - df['Grand Total'].std()

high_volume_low_perf = agent_perf_volume[
    (agent_perf_volume['Call Volume'] >= high_volume_threshold) &
    (agent_perf_volume['Grand Total'] <= low_perf_threshold)
]

if len(high_volume_low_perf) > 0:
    risk_factors.append(f"🚨 **High-Risk Agents**: {len(high_volume_low_perf)} agents handling high volume but low performance")

# Training backlog
training_issues_count = df['Training Issues'].apply(len).sum()
if training_issues_count > len(df) * 0.2:
    risk_factors.append(f"📚 **Training Backlog**: {training_issues_count} training issues identified across {len(df)} calls")

# Sentiment deterioration
negative_calls = (df['Summary Sentiment'] == 'Negative').sum()
if negative_calls > len(df) * 0.15:
    risk_factors.append(f"😞 **Customer Satisfaction Risk**: {negative_calls} negative sentiment calls ({negative_calls/len(df)*100:.1f}%)")

if risk_factors:
    st.error("### Critical Business Risks Identified")
    for risk in risk_factors:
        st.write(risk)
else:
    st.success("✅ No critical business risks detected")

# Strategic Recommendations
st.subheader("🎯 Strategic Recommendations")

recommendations = []

# Based on performance analysis
top_performer = agent_perf_volume['Grand Total'].idxmax()
top_score = agent_perf_volume['Grand Total'].max()
recommendations.append(f"🏆 **Best Practice Replication**: Study {top_performer}'s approach (Score: {top_score:.1f}) and create training modules")

# Based on training needs
if training_issues_count > 0:
    top_issue = pd.Series([issue for issues in df['Training Issues'] for issue in issues]).value_counts().index[0]
    recommendations.append(f"🎓 **Priority Training Focus**: Address '{top_issue}' issues affecting {pd.Series([issue for issues in df['Training Issues'] for issue in issues]).value_counts()[0]} calls")

# Based on workload distribution
workload_cv = workload_std / avg_workload  # Coefficient of variation
if workload_cv > 0.3:
    recommendations.append(f"⚖️ **Workload Balancing**: High workload variation (CV: {workload_cv:.2f}) suggests need for better call distribution")

# Based on performance trends
if not df['Date of the call'].isna().all():
    recent_performance = df[df['Date of the call'] >= df['Date of the call'].max() - pd.Timedelta(days=30)]['Grand Total'].mean()
    overall_performance = df['Grand Total'].mean()
    if recent_performance < overall_performance * 0.95:
        recommendations.append(f"📉 **Performance Trending Down**: Recent scores ({recent_performance:.1f}) below overall average ({overall_performance:.1f})")

for rec in recommendations:
    st.info(rec)

# Executive Summary Export
st.subheader("📊 Executive Summary Report")

if st.button("Generate Executive Summary"):
    summary_data = {
        "Metric": [
            "Total Calls Analyzed",
            "Average Call Score",
            "Agents Active",
            "Excellent Performance Rate",
            "Training Issues Identified",
            "Negative Sentiment Calls",
            "Projected ROI from Training",
            "High-Risk Agents"
        ],
        "Value": [
            len(df),
            f"{df['Grand Total'].mean():.1f}/100",
            df['Agent name'].nunique(),
            f"{(df['Performance Category'] == 'Excellent').mean()*100:.1f}%",
            training_issues_count,
            negative_calls,
            f"{roi:.1f}%" if roi > 0 else "N/A",
            len(high_volume_low_perf) if 'high_volume_low_perf' in locals() else 0
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Download button
    csv_summary = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Executive Summary (CSV)",
        data=csv_summary,
        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# PQR/PSR Analysis Section
st.header("📊 PQR/PSR Deep Analysis")

# Calculate PQR (Product Quality Ratio) and PSR (Process Success Rate)
st.subheader("🔢 Quality Metrics Calculation")

# PQR: Based on medication authorization success and rejection handling
clinical_calls = df[df['Dataset'] == 'Clinical'] if 'Dataset' in df.columns else df

if len(clinical_calls) > 0:
    # PQR: Successful medication authorizations vs rejections
    medication_calls = clinical_calls[clinical_calls['Query type'].str.contains('Medication|Chronic', na=False, case=False)]
    successful_auths = medication_calls[medication_calls['Grand Total'] >= 85]  # Assuming 85+ is successful
    pqr = (len(successful_auths) / len(medication_calls) * 100) if len(medication_calls) > 0 else 0

    # PSR: Process compliance and call handling efficiency
    process_compliant = clinical_calls[clinical_calls['Grand Total'] >= 75]  # Basic process compliance
    psr = (len(process_compliant) / len(clinical_calls) * 100) if len(clinical_calls) > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PQR (Product Quality Ratio)", f"{pqr:.1f}%", "Medication Auth Success")

    with col2:
        st.metric("PSR (Process Success Rate)", f"{psr:.1f}%", "Process Compliance")

    with col3:
        overall_quality = (pqr + psr) / 2
        st.metric("Overall Quality Index", f"{overall_quality:.1f}%", "Combined Score")

    # PQR/PSR Trend Analysis
    if not clinical_calls['Date of the call'].isna().all():
        clinical_calls_copy = clinical_calls.copy()
        clinical_calls_copy['Month'] = clinical_calls_copy['Date of the call'].dt.to_period('M').astype(str)

        monthly_quality = clinical_calls_copy.groupby('Month').agg({
            'Grand Total': 'mean',
            'Query type': lambda x: (x.str.contains('Medication|Chronic', na=False, case=False)).sum()
        }).reset_index()

        monthly_quality['PQR_Estimate'] = monthly_quality['Grand Total'] * 0.9  # Estimated PQR
        monthly_quality['PSR_Estimate'] = monthly_quality['Grand Total'] * 0.8  # Estimated PSR

        fig_quality_trend = px.line(
            monthly_quality,
            x='Month',
            y=['PQR_Estimate', 'PSR_Estimate'],
            title="PQR/PSR Quality Trends Over Time",
            labels={'value': 'Quality Score (%)', 'Month': 'Month'},
            markers=True
        )
        st.plotly_chart(fig_quality_trend, use_container_width=True)

# PQR/PSR by Agent
st.subheader("👥 Agent-Level PQR/PSR Analysis")

if len(clinical_calls) > 0:
    agent_quality = clinical_calls.groupby('Agent name').agg({
        'Grand Total': ['mean', 'count', 'std'],
        'Query type': lambda x: (x.str.contains('Medication|Chronic', na=False, case=False)).sum()
    }).round(2)

    agent_quality.columns = ['avg_score', 'total_calls', 'score_std', 'med_calls']
    agent_quality['PQR_Estimate'] = agent_quality['avg_score'] * 0.9
    agent_quality['PSR_Estimate'] = agent_quality['avg_score'] * 0.8
    agent_quality['Quality_Index'] = (agent_quality['PQR_Estimate'] + agent_quality['PSR_Estimate']) / 2

    # Sort by Quality Index
    agent_quality_sorted = agent_quality.sort_values('Quality_Index', ascending=True)

    fig_agent_quality = px.bar(
        agent_quality_sorted,
        x='Quality_Index',
        y=agent_quality_sorted.index,
        orientation='h',
        title="Agent Quality Index (PQR + PSR)",
        labels={'Quality_Index': 'Quality Index (%)'},
        color='Quality_Index',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_agent_quality, use_container_width=True)

    # Agent Quality Table
    st.subheader("Agent Quality Metrics Table")
    display_quality = agent_quality_sorted[['total_calls', 'med_calls', 'PQR_Estimate', 'PSR_Estimate', 'Quality_Index']].round(1)
    display_quality.columns = ['Total Calls', 'Med Calls', 'PQR (%)', 'PSR (%)', 'Quality Index (%)']
    st.dataframe(display_quality, use_container_width=True)

# PQR/PSR Impact Analysis
st.subheader("💰 Business Impact of PQR/PSR")

# Cost calculations based on quality metrics
avg_call_cost = st.number_input("Average Cost per Call (R)", value=50, min_value=0)
monthly_calls = len(clinical_calls) * 2 if len(clinical_calls) > 0 else 100  # Estimate

# Quality improvement scenarios
improvement_scenarios = {
    "Current Performance": 0,
    "5% Quality Improvement": 5,
    "10% Quality Improvement": 10,
    "15% Quality Improvement": 15
}

selected_scenario = st.selectbox("Select Quality Improvement Scenario", list(improvement_scenarios.keys()))

improvement_pct = improvement_scenarios[selected_scenario]
improved_quality = overall_quality + improvement_pct if 'overall_quality' in locals() else 70 + improvement_pct

# Calculate business impact
monthly_savings = (monthly_calls * avg_call_cost * improvement_pct / 100)
annual_savings = monthly_savings * 12

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Monthly Cost Savings", f"R{monthly_savings:,.0f}", f"{improvement_pct}% improvement")

with col2:
    st.metric("Annual Cost Savings", f"R{annual_savings:,.0f}", f"R{monthly_savings:,.0f}/month")

with col3:
    roi_period = st.number_input("ROI Period (months)", value=6, min_value=1)
    total_investment = st.number_input("Training Investment (R)", value=50000, min_value=0)
    roi = (annual_savings * (roi_period/12) / total_investment * 100) if total_investment > 0 else 0
    st.metric("ROI at Training Investment", f"{roi:.1f}%", f"{roi_period} months")

# PQR/PSR Recommendations
st.subheader("🎯 PQR/PSR Optimization Recommendations")

recommendations = []

if 'pqr' in locals() and pqr < 85:
    recommendations.append(f"**Medication Authorization Excellence**: PQR at {pqr:.1f}% - Focus on reducing medication rejections through better authorization processes")

if 'psr' in locals() and psr < 80:
    recommendations.append(f"**Process Compliance Training**: PSR at {psr:.1f}% - Implement standardized call handling procedures and compliance checklists")

if len(clinical_calls) > 0:
    low_performers = agent_quality[agent_quality['Quality_Index'] < 70].index.tolist()
    if low_performers:
        recommendations.append(f"**Targeted Agent Development**: Focus training on {', '.join(low_performers)} to improve PQR/PSR scores")

recommendations.extend([
    "**Real-time Quality Monitoring**: Implement live PQR/PSR dashboards for immediate feedback",
    "**Standardized Authorization Protocols**: Develop clear medication approval workflows",
    "**Agent Certification Program**: Regular PQR/PSR assessments with certification requirements",
    "**Quality-based Incentives**: Tie performance bonuses to PQR/PSR improvements"
])

for rec in recommendations:
    st.info(rec)

# Footer
st.markdown("---")
st.markdown("**V1 Pharma Quality Control System** - Advanced PQR/PSR analytics for pharmaceutical call center excellence")
st.markdown("*PQR (Product Quality Ratio) & PSR (Process Success Rate) drive medication authorization and process optimization*")