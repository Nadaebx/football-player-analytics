import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

st.set_page_config(page_title="Football Player Analytics", layout="wide", page_icon="⚽")

# Premium Custom CSS Injection
st.markdown("""
<style>
/* Main Background and Text */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161821;
    border-right: 1px solid #2e303e;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
h1 {
    background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 10px;
}
h2 {
    color: #00d2ff;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e222d;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 20px;
    color: #a0a0a0;
    font-weight: 500;
    border: 1px solid #2e303e;
    border-bottom: none;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #fff;
    background-color: #2a2f3d;
}

.stTabs [aria-selected="true"] {
    background-color: #2a2f3d;
    color: #00d2ff !important;
    border-top: 2px solid #00d2ff;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #00d2ff;
}

/* Dataframe header */
th {
    background-color: #1e222d !important;
    color: #00d2ff !important;
}

</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Player Analytics Dashboard")
st.markdown("A complete Data Mining deployment layer using the CRISP-DM methodology.")

@st.cache_data
def load_data():
    df = pd.read_csv('2022-2023 Football Player Stats.csv', sep=';', encoding='latin-1')
    # Data Preparation: Filter and Clean
    df = df[df['Min'] > 300].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Feature scaling for models
    drop_cols = ['Rk', 'Player', 'Nation', 'Squad', 'Comp', 'Born', 'Pos', 'Goals', 'G/Sh', 'G/SoT', 'GCA', 'GcaPassLive', 'GcaPassDead', 'GcaDrib', 'GcaSh', 'GcaFld', 'GcaDef']
    X_full = df.drop(columns=drop_cols, errors='ignore')
    features = X_full.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    return df, X_scaled, features

df, X_scaled, features = load_data()

# GLOBAL FILTER
st.sidebar.header("🌍 Global Filters")
leagues = ['All'] + list(df['Comp'].unique())
selected_league = st.sidebar.selectbox("Select League (Comp)", leagues)

if selected_league != 'All':
    mask = df['Comp'] == selected_league
    filtered_df = df[mask].copy()
    filtered_indices = df.index[mask].tolist()
    # Need to match scaled data to filtered indices
    X_filtered = X_scaled[mask]
else:
    filtered_df = df.copy()
    X_filtered = X_scaled

# Top Level Metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Players Analyzed", len(filtered_df))
col2.metric("Statistical Features", len(features))
col3.metric("Selected League", selected_league.replace('eng ', '').title() if selected_league != 'All' else 'All')
col4.metric("Avg Minutes Played", f"{int(filtered_df['Min'].mean())} mins")
st.markdown("---")

tabs = st.tabs(["📊 Business Insights", "🎯 Classification (Scorer Class)", "🧠 Clustering (Playstyles)", "📈 Regression (Goals)", "🗺️ PCA (Mapping)"])

with tabs[0]:
    st.header("Business Insights")
    st.markdown("""
    ### Top 3 Actionable Findings
    
    ✨ **1. Playstyle > Position:**  
    Modern football roles are highly fluid. Our clustering algorithm reveals that players listed under traditional positions often share identical statistical profiles with other positions (e.g., Attacking Fullbacks cluster with Wingers).
    
    🎯 **2. Creation Equals Output:**  
    The regression analysis shows that Shot-Creating Actions (SCA) strongly predict goal output. Clubs can scout undervalued players by finding those with high SCA who are currently underperforming their expected goals.
    
    🗺️ **3. Dimensionality Reduction for Scouting:**  
    The PCA map reduces over 100 complex metrics into a 2D space. Scouts can instantly identify "hidden gems" by finding lesser-known players who map closely to established superstars.
    """)
    st.info("💡 Navigate through the tabs above to explore the interactive models.")

with tabs[1]:
    st.header("Classification: Predicting Scorer Class")
    st.write("Using a **Random Forest Classifier** to predict if a player is a High or Low Scorer.")
    
    # Train model on full data for consistency, then display filtered
    median_goals = filtered_df['Goals'].median()
    filtered_df['Scorer_Class'] = np.where(filtered_df['Goals'] >= median_goals, 'High Scorer', 'Low Scorer')
    
    if len(filtered_df) > 10:
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_filtered, filtered_df['Scorer_Class'])
        filtered_df['Predicted_Class'] = clf.predict(X_filtered)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.histogram(filtered_df[['Scorer_Class', 'Predicted_Class']], x="Scorer_Class", color="Predicted_Class", barmode="group",
                               title="Actual vs Predicted Scorer Class",
                               labels={"Scorer_Class": "Actual Class"},
                               color_discrete_sequence=['#00d2ff', '#ff4b4b'],
                               template="plotly_dark")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            
        with col2:
            st.write("### 🔍 Misclassified Players")
            misclassified = filtered_df[filtered_df['Scorer_Class'] != filtered_df['Predicted_Class']]
            st.dataframe(misclassified[['Player', 'Squad', 'Scorer_Class', 'Predicted_Class', 'Min']].head(10), width="stretch")
    else:
        st.warning("Not enough data to run Classification in this filter.")

with tabs[2]:
    st.header("Clustering: K-Means Playstyle Profiles")
    st.write("Grouping players into 4 distinct playstyle profiles based on their 120+ statistical features.")
    
    if len(filtered_df) > 10:
        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
        filtered_df['Cluster'] = kmeans.fit_predict(X_filtered)
        filtered_df['Cluster'] = "Profile " + filtered_df['Cluster'].astype(str)
        
        fig2 = px.scatter(filtered_df[['Player', 'PasProg', 'TklWon', 'Cluster', 'Squad', 'Pos', 'Min']], x="PasProg", y="TklWon", color="Cluster", hover_name="Player",
                          hover_data=["Squad", "Pos", "Min"],
                          title="Progressive Passes vs Tackles Won (Colored by Cluster)",
                          labels={"PasProg": "Progressive Passes", "TklWon": "Tackles Won"},
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          template="plotly_dark")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig2.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig2, width="stretch")
        
        st.caption("✨ Note how clusters separate players not just by position, but by their active contribution (e.g., defensive output vs ball progression).")

with tabs[3]:
    st.header("Regression: Predicting Goals")
    st.write("Using **Linear Regression** to predict actual `Goals` scored based on Shot-Creating Actions (`SCA`).")
    
    if len(filtered_df) > 10:
        X_reg = filtered_df[['SCA']].values
        y_reg = filtered_df['Goals'].values
        
        reg = LinearRegression()
        reg.fit(X_reg, y_reg)
        filtered_df['Predicted_Goals'] = reg.predict(X_reg)
        
        fig3 = px.scatter(filtered_df[['Player', 'SCA', 'Goals', 'Squad', 'Min']], x="SCA", y="Goals", hover_name="Player", hover_data=["Squad", "Min"],
                          title="Actual Goals vs Shot-Creating Actions (SCA)",
                          trendline="ols",
                          trendline_color_override="#ff4b4b",
                          color="Goals",
                          color_continuous_scale="Blues",
                          template="plotly_dark")
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig3.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig3, width="stretch")
        st.caption("📈 Players significantly *above* the trendline are highly clinical (or overperforming). Players *below* the line are creating many chances but not scoring them.")

with tabs[4]:
    st.header("Dimensionality Reduction: PCA Map")
    st.write("Reducing 120+ features into 2 dimensions to visualize the entire player landscape.")
    
    if len(filtered_df) > 10:
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_filtered)
        
        filtered_df['PC1'] = components[:, 0]
        filtered_df['PC2'] = components[:, 1]
        
        fig4 = px.scatter(filtered_df[['Player', 'PC1', 'PC2', 'Pos', 'Squad', 'Comp']], x="PC1", y="PC2", color="Pos", hover_name="Player",
                          hover_data=["Squad", "Comp"],
                          title="PCA 2D Mapping of European Football Players",
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          template="plotly_dark")
        fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig4.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
        st.plotly_chart(fig4, width="stretch")
        st.write("**PC1 (X-Axis):** Generally separates Defending/Possession from Attacking/Creativity.")
        st.write("**PC2 (Y-Axis):** Generally captures volume of involvement (e.g., high-touch midfielders vs low-touch strikers).")
