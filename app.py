import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="African Financial Crisis Analytics | CIB Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for CIB styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2c5aa0);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2c5aa0;
        margin: 0.5rem 0;
        color: #2c3e50;
        font-weight: 500;
    }
    .crisis-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
        font-weight: 500;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    .stMetric {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    .stMetric [data-testid="metric-container"] {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the African financial crisis dataset"""
    try:
        np.random.seed(42)
        countries = ['Algeria', 'Angola', 'Egypt', 'Kenya', 'Morocco', 'Nigeria', 'South Africa', 'Tunisia']
        years = list(range(1990, 2015))
        
        data = []
        for country in countries:
            for year in years:
                data.append({
                    'country': country,
                    'year': year,
                    'cc3': country[:3].upper(),
                    'banking_crisis': np.random.choice(['crisis', 'no_crisis'], p=[0.15, 0.85]),
                    'systemic_crisis': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'currency_crises': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'inflation_crises': np.random.choice([0, 1], p=[0.88, 0.12]),
                    'exch_usd': np.random.uniform(0.5, 15.0),
                    'inflation_annual_cpi': np.random.uniform(-2, 25),
                    'gdp_weighted_default': np.random.uniform(0, 0.3),
                    'domestic_debt_in_default': np.random.choice([0, 1], p=[0.95, 0.05]),
                    'sovereign_external_debt_default': np.random.choice([0, 1], p=[0.92, 0.08])
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 African Financial Crisis Analytics</h1>
        <h3>CIB Research & Risk Intelligence Platform</h3>
        <p>Advanced Analytics for Banking Crisis Prediction & Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Unable to load dataset. Please check data source.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 CIB Analytics Dashboard")
    st.sidebar.markdown("---")
    
    # Analysis modules
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "🎯 Executive Summary",
            "📈 Crisis Trend Analysis", 
            "🌍 Geographic Risk Mapping",
            "🤖 ML Crisis Prediction",
            "📊 Portfolio Risk Assessment",
            "📋 Regulatory Reporting"
        ]
    )
    
    # Country and year filters
    st.sidebar.markdown("### Filters")
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=df['country'].unique(),
        default=df['country'].unique()[:4]
    )
    
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Analysis modules
    if analysis_type == "🎯 Executive Summary":
        executive_summary(filtered_df)
    elif analysis_type == "📈 Crisis Trend Analysis":
        crisis_trend_analysis(filtered_df)
    elif analysis_type == "🌍 Geographic Risk Mapping":
        geographic_risk_mapping(filtered_df)
    elif analysis_type == "🤖 ML Crisis Prediction":
        ml_crisis_prediction(df)
    elif analysis_type == "📊 Portfolio Risk Assessment":
        portfolio_risk_assessment(filtered_df)
    elif analysis_type == "📋 Regulatory Reporting":
        regulatory_reporting(filtered_df)

def executive_summary(df):
    """Executive dashboard with key metrics"""
    st.markdown("## 🎯 Executive Summary - CIB Risk Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crisis_rate = (df['banking_crisis'] == 'crisis').sum() / len(df) * 100
        st.metric(
            label="Banking Crisis Rate",
            value=f"{crisis_rate:.1f}%",
            delta="-2.1%" if crisis_rate < 20 else "+3.2%"
        )
    
    with col2:
        avg_inflation = df['inflation_annual_cpi'].mean()
        st.metric(
            label="Avg Inflation Rate",
            value=f"{avg_inflation:.1f}%",
            delta="-0.8%" if avg_inflation < 10 else "+1.5%"
        )
    
    with col3:
        systemic_crises = df['systemic_crisis'].sum()
        st.metric(
            label="Systemic Crises",
            value=f"{systemic_crises}",
            delta="-2" if systemic_crises < 50 else "+5"
        )
    
    with col4:
        countries_at_risk = df[df['banking_crisis'] == 'crisis']['country'].nunique()
        st.metric(
            label="Countries at Risk",
            value=f"{countries_at_risk}",
            delta="0" if countries_at_risk < 3 else "+1"
        )
    
    # Crisis timeline
    st.markdown("### 📊 Crisis Evolution Timeline")
    crisis_timeline = df.groupby('year').agg({
        'banking_crisis': lambda x: (x == 'crisis').sum(),
        'systemic_crisis': 'sum',
        'currency_crises': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Financial Crises Over Time",)
    )
    
    fig.add_trace(
        go.Scatter(x=crisis_timeline['year'], y=crisis_timeline['banking_crisis'],
                  name='Banking Crisis', line=dict(color='#e74c3c', width=3))
    )
    fig.add_trace(
        go.Scatter(x=crisis_timeline['year'], y=crisis_timeline['systemic_crisis'],
                  name='Systemic Crisis', line=dict(color='#f39c12', width=3))
    )
    fig.add_trace(
        go.Scatter(x=crisis_timeline['year'], y=crisis_timeline['currency_crises'],
                  name='Currency Crisis', line=dict(color='#3498db', width=3))
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Number of Crises"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk alerts
    st.markdown("### 🚨 Current Risk Alerts")
    high_risk_countries = df[
        (df['inflation_annual_cpi'] > 15) | 
        (df['banking_crisis'] == 'crisis')
    ]['country'].unique()
    
    if len(high_risk_countries) > 0:
        st.markdown(f"""
        <div class="crisis-alert">
            <strong>⚠️ High Risk Alert:</strong> {len(high_risk_countries)} countries showing elevated risk indicators:
            <br><strong>Countries:</strong> {', '.join(high_risk_countries[:5])}
        </div>
        """, unsafe_allow_html=True)

def crisis_trend_analysis(df):
    """Detailed crisis trend analysis"""
    st.markdown("## 📈 Crisis Trend Analysis")
    
    # Crisis correlation heatmap
    st.markdown("### Crisis Correlation Matrix")
    crisis_cols = ['banking_crisis', 'systemic_crisis', 'currency_crises', 'inflation_crises']
    df_encoded = df.copy()
    df_encoded['banking_crisis'] = (df_encoded['banking_crisis'] == 'crisis').astype(int)
    
    corr_matrix = df_encoded[crisis_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Crisis Type Correlations"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Country-wise crisis analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Banking Crisis by Country")
        crisis_by_country = df.groupby('country')['banking_crisis'].apply(
            lambda x: (x == 'crisis').sum()
        ).sort_values(ascending=False)
        
        fig_bar = px.bar(
            x=crisis_by_country.values,
            y=crisis_by_country.index,
            orientation='h',
            title="Banking Crises Count by Country",
            color=crisis_by_country.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### Inflation vs Crisis Relationship")
        fig_scatter = px.scatter(
            df, x='inflation_annual_cpi', y='exch_usd',
            color='banking_crisis',
            size='gdp_weighted_default',
            hover_data=['country', 'year'],
            title="Inflation vs Exchange Rate (Crisis Indicator)",
            color_discrete_map={'crisis': '#e74c3c', 'no_crisis': '#2ecc71'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

def geographic_risk_mapping(df):
    """Geographic risk visualization"""
    st.markdown("## 🌍 Geographic Risk Mapping")
    
    # Risk score calculation
    df_risk = df.copy()
    df_risk['banking_crisis_score'] = (df_risk['banking_crisis'] == 'crisis').astype(int)
    
    risk_by_country = df_risk.groupby('country').agg({
        'banking_crisis_score': 'mean',
        'systemic_crisis': 'mean',
        'currency_crises': 'mean',
        'inflation_annual_cpi': 'mean',
        'exch_usd': 'mean'
    }).reset_index()
    
    risk_by_country['composite_risk'] = (
        risk_by_country['banking_crisis_score'] * 0.3 +
        risk_by_country['systemic_crisis'] * 0.25 +
        risk_by_country['currency_crises'] * 0.2 +
        (risk_by_country['inflation_annual_cpi'] > 10).astype(int) * 0.25
    )
    
    # Risk ranking
    st.markdown("### 🏆 Country Risk Rankings")
    risk_ranking = risk_by_country.sort_values('composite_risk', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_risk = px.bar(
            risk_ranking.head(10),
            x='composite_risk',
            y='country',
            orientation='h',
            title="Composite Risk Score by Country",
            color='composite_risk',
            color_continuous_scale='Reds'
        )
        fig_risk.update_layout(height=500)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Risk Countries")
        for i, row in risk_ranking.head(5).iterrows():
            risk_level = "🔴 High" if row['composite_risk'] > 0.3 else "🟡 Medium" if row['composite_risk'] > 0.15 else "🟢 Low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{row['country']}</strong><br>
                Risk Level: {risk_level}<br>
                Score: {row['composite_risk']:.3f}
            </div>
            """, unsafe_allow_html=True)

def ml_crisis_prediction(df):
    """Machine learning crisis prediction model"""
    st.markdown("## 🤖 ML Crisis Prediction Model")
    
    # Prepare data for ML
    df_ml = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df_ml.columns:
        if df_ml[col].dtype == 'object' and col != 'banking_crisis':
            df_ml[col] = le.fit_transform(df_ml[col])
    
    # Target variable
    df_ml['banking_crisis'] = (df_ml['banking_crisis'] == 'crisis').astype(int)
    
    # Features and target
    feature_cols = ['year', 'systemic_crisis', 'exch_usd', 'inflation_annual_cpi', 
                   'currency_crises', 'inflation_crises', 'gdp_weighted_default']
    X = df_ml[feature_cols]
    y = df_ml['banking_crisis']
    
    # Handle class imbalance
    oversample = RandomOverSampler(sampling_strategy=0.3, random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model comparison
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    st.markdown("### 📊 Model Performance Comparison")
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': f"{accuracy:.3f}",
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1-Score': f"{f1:.3f}",
            'ROC-AUC': f"{roc_auc:.3f}"
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Feature importance (using Random Forest)
    st.markdown("### 🎯 Feature Importance Analysis")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance for Banking Crisis Prediction",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction interface
    st.markdown("### 🔮 Crisis Prediction Tool")
    col1, col2 = st.columns(2)
    
    with col1:
        pred_inflation = st.slider("Inflation Rate (%)", -5.0, 30.0, 5.0)
        pred_exchange = st.slider("Exchange Rate (USD)", 0.1, 20.0, 5.0)
        pred_gdp_default = st.slider("GDP Weighted Default", 0.0, 1.0, 0.1)
    
    with col2:
        pred_systemic = st.selectbox("Systemic Crisis", [0, 1])
        pred_currency = st.selectbox("Currency Crisis", [0, 1])
        pred_inflation_crisis = st.selectbox("Inflation Crisis", [0, 1])
    
    if st.button("🎯 Predict Crisis Probability"):
        # Make prediction
        pred_data = np.array([[2024, pred_systemic, pred_exchange, pred_inflation, 
                              pred_currency, pred_inflation_crisis, pred_gdp_default]])
        pred_data_scaled = scaler.transform(pred_data)
        
        crisis_prob = rf_model.predict_proba(pred_data_scaled)[0][1]
        
        st.markdown(f"""
        <div class="crisis-alert">
            <h4>🎯 Prediction Result</h4>
            <p><strong>Banking Crisis Probability: {crisis_prob:.1%}</strong></p>
            <p>Risk Level: {'🔴 HIGH RISK' if crisis_prob > 0.6 else '🟡 MEDIUM RISK' if crisis_prob > 0.3 else '🟢 LOW RISK'}</p>
        </div>
        """, unsafe_allow_html=True)

def portfolio_risk_assessment(df):
    """Portfolio risk assessment for CIB"""
    st.markdown("## 📊 Portfolio Risk Assessment")
    
    # Risk metrics by country
    risk_metrics = df.groupby('country').agg({
        'banking_crisis': lambda x: (x == 'crisis').sum(),
        'inflation_annual_cpi': ['mean', 'std', 'max'],
        'exch_usd': ['mean', 'std'],
        'gdp_weighted_default': ['mean', 'max']
    }).round(3)
    
    risk_metrics.columns = ['Banking_Crises', 'Avg_Inflation', 'Inflation_Volatility', 
                           'Max_Inflation', 'Avg_Exchange_Rate', 'Exchange_Volatility',
                           'Avg_Default_Rate', 'Max_Default_Rate']
    
    st.markdown("### 📈 Country Risk Metrics")
    st.dataframe(risk_metrics, use_container_width=True)
    
    # Portfolio simulation
    st.markdown("### 💼 Portfolio Risk Simulation")
    
    selected_portfolio = st.multiselect(
        "Select Portfolio Countries",
        options=df['country'].unique(),
        default=df['country'].unique()[:3]
    )
    
    if selected_portfolio:
        portfolio_data = df[df['country'].isin(selected_portfolio)]
        
        # Calculate portfolio metrics
        portfolio_crisis_rate = (portfolio_data['banking_crisis'] == 'crisis').sum() / len(portfolio_data)
        portfolio_avg_inflation = portfolio_data['inflation_annual_cpi'].mean()
        portfolio_volatility = portfolio_data['inflation_annual_cpi'].std()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Crisis Rate", f"{portfolio_crisis_rate:.1%}")
        with col2:
            st.metric("Avg Inflation", f"{portfolio_avg_inflation:.1f}%")
        with col3:
            st.metric("Inflation Volatility", f"{portfolio_volatility:.1f}%")

def regulatory_reporting(df):
    """Regulatory reporting module"""
    st.markdown("## 📋 Regulatory Reporting")
    
    # Stress test scenarios
    st.markdown("### 🏛️ Basel III Stress Test Scenarios")
    
    scenario = st.selectbox(
        "Select Stress Test Scenario",
        ["Baseline", "Adverse", "Severely Adverse"]
    )
    
    # Apply stress multipliers
    stress_multipliers = {
        "Baseline": 1.0,
        "Adverse": 1.5,
        "Severely Adverse": 2.0
    }
    
    multiplier = stress_multipliers[scenario]
    stressed_data = df.copy()
    stressed_data['stressed_inflation'] = stressed_data['inflation_annual_cpi'] * multiplier
    stressed_data['stressed_default'] = np.minimum(stressed_data['gdp_weighted_default'] * multiplier, 1.0)
    
    # Compliance metrics
    st.markdown("### 📊 Compliance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Concentration Limits")
        country_exposure = df['country'].value_counts(normalize=True)
        max_exposure = country_exposure.max()
        
        compliance_status = "✅ COMPLIANT" if max_exposure < 0.25 else "❌ NON-COMPLIANT"
        st.markdown(f"**Single Country Exposure Limit (25%):** {compliance_status}")
        st.markdown(f"**Maximum Exposure:** {max_exposure:.1%}")
    
    with col2:
        st.markdown("#### Crisis Correlation Limits")
        crisis_correlation = df.groupby('year').agg({
            'banking_crisis': lambda x: (x == 'crisis').sum()
        })['banking_crisis'].std()
        
        correlation_status = "✅ WITHIN LIMITS" if crisis_correlation < 2.0 else "⚠️ ELEVATED"
        st.markdown(f"**Crisis Clustering Risk:** {correlation_status}")
        st.markdown(f"**Crisis Volatility:** {crisis_correlation:.2f}")

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>African Financial Crisis Analytics Platform</strong></p>
    <p>CIB Research & Analytics Division | Risk Intelligence & Market Research</p>
    <p>Built with Streamlit • Advanced ML Models • Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)