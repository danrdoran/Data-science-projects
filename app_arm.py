import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings, random

# ════════════════════════════════════════
# PAGE CONFIG & STYLE
# ════════════════════════════════════════
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

st.set_page_config(page_title="Arab Risk Monitor - Multidimensional Risk Analysis", layout="wide")
PURPLE = "#FBF8FF"
st.markdown(f"""
    <style>
        .stApp, [data-testid="stAppViewContainer"] {{ background: {PURPLE}; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {PURPLE}; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        .metric-label {{ font-size: 14px !important; }}
        .metric-value {{ font-size: 24px !important; font-weight: bold !important; }}
    </style>
""", unsafe_allow_html=True)
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"]   = "none"
plt.rcParams["savefig.facecolor"] = "none"

st.title("📈 Arab Risk Monitor - Multidimensional Risk Analysis")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ARAB_COUNTRIES = [
    'Algeria','Bahrain','Comoros','Djibouti','Egypt','Iraq','Jordan',
    'Kuwait','Lebanon','Libya','Mauritania','Morocco','Oman','Palestine, State of',
    'Qatar','Saudi Arabia','Somalia','Sudan','Syrian Arab Republic',
    'Tunisia','United Arab Emirates','Yemen'
]



# ════════════════════════════════════════
# Helper: readable labels
# ════════════════════════════════════════
def make_readable_name(feature_name: str) -> str:
    readable_names = {
        '01-fatper_nm': 'Fatalities per 100,000 Population',
        '02-political_stability_nm': 'Political Stability',
        '03-conflict proximity_nm': 'Proximity to Other Conflicts',
        '04-state_authority_nm': 'State Authority over Territory',
        '05-disp_rate_nm': 'Population Displacement Rate',
        '06-voice_accountability_nm': 'Voice & Accountability',
        '7-ren_water_nm': 'Renewable Water Resources per Person',
        '8-wather_withd_nm': 'Water Withdrawal Rate',
        '9-pop_dis_nm': 'Internally Displaced Population',
        '10-dis_dicp_nm': 'Disaster-Related Displacement',
        '11-adap_strat_nm': 'Climate Adaptation Strategies',
        '12-cli_fin_nm': 'Climate Finance Received',
        '13-remit_nm': 'Remittances (% of GDP)',
        '14-oda_nm': 'Official Development Assistance (% of GNI)',
        '15-food_ins_nm': 'Food Insecurity Level',
        '16-undernour_nm': 'Undernourishment Rate',
        '17-gini_nm': 'Income Inequality (Gini Index)',
        '18-topbottom_ratio_nm': 'Top-to-Bottom Income Ratio',
        '19-govt_debt_nm': 'Government Debt (% of GDP)',
        '20-int_use_nm': 'Internet Usage Rate',
        '21-unemp_nm': 'Unemployment Rate',
        '22-youth_nm': 'Youth Population Share',
        '23-mr5_nm': 'Under-5 Mortality Rate',
        '24-mmr_nm': 'Maternal Mortality Ratio',
        '25-exp_sch_nm': 'Expected Years of Schooling',
        '26-mean_sch_nm': 'Mean Years of Schooling',
        '27-soc_pro_nm': 'Social Protection Coverage',
        '28-water_serv_nm': 'Access to Improved Water Sources',
        '29-san_serv_nm': 'Access to Sanitation Services',
        '30-uhc_nm': 'Universal Health Coverage Index',
        '31-gpi_nm': 'Global Peace Index Score',
        '32-lfp_fem_nm': 'Female Labor Force Participation',
        '33-control_corruption_nm': 'Control of Corruption',
        '34-rule_law_nm': 'Rule of Law',
        '35-government_eff_nm': 'Government Effectiveness',
        '36-osi_nm': 'Online Services Index',
        '37-emp_to_pop_nm': 'Employment-to-Population Ratio',
        '38-territory_control_nm': 'Territorial Control',
        '39-H_index_nm': 'H-index of Citable Documents',
        '40-health_exp_nm': 'Health Expenditure (% of GDP)',
        '41-mean_schooling_nm': 'Mean Schooling Years (Alt Source)',
        '42-refugees_per_100k_nm': 'Refugees per 100,000 Population',
        '43-out_of_school_nm': 'Out-of-School Children',
        '44-precipitation_nm': 'Average Annual Precipitation',
        '45-rd_gdp_nm': 'R&D Expenditure (% of GDP)',
        '46-tax_gdp_nm': 'Tax Revenue (% of GDP)',
        '47-tech_dependence_nm': 'Technological Dependence Index',
        '48-water_stress_nm': 'Water Stress Level',
        '49-debt_gni_nm': 'Debt-to-GNI Ratio',
        '50-humanitarian_aid_nm': 'Humanitarian Aid Received',
        '51-agri_land_nm': 'Agricultural Land Area (% of Total Land)',
        '52-gdp_ppp_nm': 'GDP per Capita (PPP, USD)',
        '53-agri_gdp_nm': 'Agriculture Share of GDP (%)',
        '54-vdem_political_pol_nm': 'Political Polarization',
        '55-vdem_discussion_nm': 'Public Discussion Quality',
        '56- vdem_participation_nm': 'Political Participation'
    }
    return readable_names.get(feature_name, feature_name.replace('_nm','').replace('_',' ').title())

# ════════════════════════════════════════
# INDICATOR HIERARCHY
# ════════════════════════════════════════
HIERARCHY: Dict[str, Dict[str, str]] = {
    # Conflict Pathway
    '01-fatper_nm': {'Pathway':'Conflict','Domain':'Conflict','Component':'Vulnerability','Driver':'Conflict Intensity'},
    '02-political_stability_nm': {'Pathway':'Conflict','Domain':'Conflict','Component':'Vulnerability','Driver':'Political Instability'},
    '03-conflict proximity_nm': {'Pathway':'Conflict','Domain':'Conflict','Component':'Vulnerability','Driver':'Neighboring Conflict'},
    '42-refugees_per_100k_nm': {'Pathway':'Conflict','Domain':'Conflict','Component':'Vulnerability','Driver':'Displacement'},
    '04-state_authority_nm': {'Pathway':'Conflict','Domain':'Conflict','Component':'Resilience','Driver':'Territorial Integrity'},

    # Climate – Climate Hazards
    '10-dis_dicp_nm': {'Pathway':'Climate','Domain':'Climate Hazards','Component':'Vulnerability','Driver':'Disaster Impact'},
    '11-adap_strat_nm': {'Pathway':'Climate','Domain':'Climate Hazards','Component':'Resilience','Driver':'Adaptation Strategies'},

    # Climate – Natural Resources
    '51-agri_land_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Vulnerability','Driver':'Land Vulnerability'},
    '53-agri_gdp_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Vulnerability','Driver':'Land Vulnerability'},
    '8-wather_withd_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Vulnerability','Driver':'Water Scarcity'},
    '48-water_stress_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Vulnerability','Driver':'Water Scarcity'},
    '7-ren_water_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Resilience','Driver':'Water Resilience'},
    '44-precipitation_nm': {'Pathway':'Climate','Domain':'Natural Resources','Component':'Resilience','Driver':'Water Resilience'},

    # Development – Economy
    '13-remit_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Financial Dependence'},
    '14-oda_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Financial Dependence'},
    '50-humanitarian_aid_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Financial Dependence'},
    '15-food_ins_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Food Insecurity'},
    '16-undernour_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Food Insecurity'},
    '17-gini_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Income Inequality'},
    '18-topbottom_ratio_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Income Inequality'},
    '19-govt_debt_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Fiscal Space'},
    '49-debt_gni_nm': {'Pathway':'Development','Domain':'Economy','Component':'Vulnerability','Driver':'Fiscal Space'},
    '52-gdp_ppp_nm': {'Pathway':'Development','Domain':'Economy','Component':'Resilience','Driver':'Economic Development'},
    '37-emp_to_pop_nm': {'Pathway':'Development','Domain':'Economy','Component':'Resilience','Driver':'Labor Market Flexibility'},
    '21-unemp_nm': {'Pathway':'Development','Domain':'Economy','Component':'Resilience','Driver':'Labor Market Flexibility'},
    '46-tax_gdp_nm': {'Pathway':'Development','Domain':'Economy','Component':'Resilience','Driver':'Revenue Resilience'},

    # Development – Social
    '22-youth_nm': {'Pathway':'Development','Domain':'Social','Component':'Vulnerability','Driver':'Youth Bulge'},
    '23-mr5_nm': {'Pathway':'Development','Domain':'Social','Component':'Vulnerability','Driver':'Infant Mortality'},
    '24-mmr_nm': {'Pathway':'Development','Domain':'Social','Component':'Vulnerability','Driver':'Maternal Mortality'},
    '43-out_of_school_nm': {'Pathway':'Development','Domain':'Social','Component':'Vulnerability','Driver':'Education Exclusion'},
    '25-exp_sch_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Education Resilience'},
    '26-mean_sch_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Education Resilience'},
    '27-soc_pro_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Social Protection'},
    '28-water_serv_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Water & Sanitation Services'},
    '29-san_serv_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Water & Sanitation Services'},
    '30-uhc_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Health Coverage'},
    '32-lfp_fem_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Female Labor Force'},
    '40-health_exp_nm': {'Pathway':'Development','Domain':'Social','Component':'Resilience','Driver':'Health Expenditure'},

    # Development – Institutions
    '33-control_corruption_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Vulnerability','Driver':'Corruption'},
    '34-rule_law_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Vulnerability','Driver':'Rule of Law'},
    '35-government_eff_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Government Effectiveness'},
    '36-osi_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Online Services Index'},
    '54-vdem_political_pol_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Political Culture'},
    '55-vdem_discussion_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Political Culture'},
    '56- vdem_participation_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Political Culture'},
    '06-voice_accountability_nm': {'Pathway':'Development','Domain':'Institutions','Component':'Resilience','Driver':'Political Culture'},

    # Development – Technology & Innovation
    '47-tech_dependence_nm': {'Pathway':'Development','Domain':'Technology & Innovation','Component':'Vulnerability','Driver':'Technology Dependence'},
    '45-rd_gdp_nm': {'Pathway':'Development','Domain':'Technology & Innovation','Component':'Resilience','Driver':'R&D'},
    '39-H_index_nm': {'Pathway':'Development','Domain':'Technology & Innovation','Component':'Resilience','Driver':'Research Impact'},
    '20-int_use_nm': {'Pathway':'Development','Domain':'Technology & Innovation','Component':'Resilience','Driver':'Internet Usage'},
}

# ════════════════════════════════════════
# EXPLICIT "HIGH IS GOOD" vs "HIGH IS BAD" LISTS
# ════════════════════════════════════════
GOOD_HIGH_FEATURES = {
    '04-state_authority_nm',
    '11-adap_strat_nm',
    '7-ren_water_nm', '44-precipitation_nm',
    '52-gdp_ppp_nm',
    '37-emp_to_pop_nm',
    '46-tax_gdp_nm',
    '20-int_use_nm',
    '25-exp_sch_nm', '26-mean_sch_nm', '41-mean_schooling_nm',
    '27-soc_pro_nm',
    '28-water_serv_nm', '29-san_serv_nm', '30-uhc_nm',
    '32-lfp_fem_nm',
    '40-health_exp_nm',
    '35-government_eff_nm','36-osi_nm',
    '33-control_corruption_nm',
    '34-rule_law_nm',
    '55-vdem_discussion_nm','56- vdem_participation_nm','06-voice_accountability_nm',
    '45-rd_gdp_nm','39-H_index_nm',
    '02-political_stability_nm',
    '12-cli_fin_nm',  # Climate finance is good when high
}

BAD_HIGH_FEATURES = {
    '01-fatper_nm','03-conflict proximity_nm','42-refugees_per_100k_nm','05-disp_rate_nm','09-pop_dis_nm',
    '10-dis_dicp_nm',
    '51-agri_land_nm','53-agri_gdp_nm',  # High agriculture dependence is vulnerability
    '8-wather_withd_nm','48-water_stress_nm',
    '13-remit_nm','14-oda_nm','50-humanitarian_aid_nm',
    '15-food_ins_nm','16-undernour_nm',
    '17-gini_nm','18-topbottom_ratio_nm',
    '19-govt_debt_nm','49-debt_gni_nm',
    '21-unemp_nm',
    '22-youth_nm',
    '23-mr5_nm','24-mmr_nm',
    '43-out_of_school_nm',
    '31-gpi_nm',
    '47-tech_dependence_nm',
    '54-vdem_political_pol_nm',
}

# ════════════════════════════════════════
# LOAD RAW
# ════════════════════════════════════════
@st.cache_data
def load_raw() -> pd.DataFrame:
    """
    Load indicator data from the Excel file only.
    No conflict CSV is used anymore.
    """
    df_ind = pd.read_excel("data/data_risk_v2.xlsx", sheet_name="data_indicators")
    df_ind["years"] = df_ind["years"].astype(int)

    # Keep only years ≥ 2006 as before
    df = df_ind[df_ind["years"] >= 2006].copy()
    return df

raw_df = load_raw()
meta_cols = ['country_code', 'country_name', 'years']
value_cols = [c for c in raw_df.columns if c not in meta_cols]
raw_df[value_cols] = raw_df[value_cols].apply(pd.to_numeric, errors='coerce')
candidate_cols = [c for c in raw_df.columns if c not in meta_cols]

# ════════════════════════════════════════
# TIME-SAFE PANEL PREP (kept for future tabs)
# ════════════════════════════════════════
def forward_fill_panel(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.sort_values(['country_name','years']).copy()
    df2[cols] = df2.groupby('country_name')[cols].ffill()
    return df2

@st.cache_data
def build_time_safe_datasets(missing_thresh: float = 0.95):
    """
    Simplified version:
    - forward-fill indicator values over time
    - keep only Arab countries
    - return the panel used by the UI
    """
    ff_df = forward_fill_panel(raw_df, candidate_cols)

    arab_mask_all = ff_df["country_name"].isin(ARAB_COUNTRIES)
    arab_df = ff_df.loc[
        arab_mask_all,
        ["country_code", "country_name", "years"] + candidate_cols
    ].copy()

    return {
        "keep_cols": candidate_cols,
        "pre": None,
        "splits": None,
        "arab_df": arab_df,
    }

data_bundle = build_time_safe_datasets(missing_thresh=0.95)
arab_df = data_bundle["arab_df"]

def fmt2_trim(x):
    import math
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{float(x):.2f}"

# ════════════════════════════════════════
# Helpers for hierarchy filters (tab-only)
# ════════════════════════════════════════

def get_filtered_features(pathway: str, domain: str, component: str, driver: str) -> List[str]:
    feats = []
    for feat, meta in HIERARCHY.items():
        # Only consider features that actually exist in arab_df (after filtering)
        if feat not in AVAILABLE_FEATURES:
            continue
        if pathway != "All" and meta['Pathway'] != pathway:
            continue
        if domain != "All" and meta['Domain'] != domain:
            continue
        if component != "All" and meta['Component'] != component:
            continue
        if driver != "All" and meta['Driver'] != driver:
            continue
        feats.append(feat)
    return feats

# ════════════════════════════════════════
# GLOBAL PERCENTILE MAPS (for absolute quartiles)
# ════════════════════════════════════════

def is_high_good(feature: str) -> bool:
    if feature in GOOD_HIGH_FEATURES:
        return True
    if feature in BAD_HIGH_FEATURES:
        return False
    comp = HIERARCHY.get(feature, {}).get('Component', '')
    if comp == 'Resilience':
        return True
    if comp == 'Vulnerability':
        return False
    name = make_readable_name(feature).lower()
    keywords_good = ['stability','authority','coverage','index','participation','effectiveness','control','access','schooling','tax','employment','internet','water resources','precipitation','h-index','r&d','voice']
    return any(k in name for k in keywords_good)

# helper: get the value for a feature for 2022 (or closest earlier year)
def get_value_for_target_year(country_df: pd.DataFrame, feature: str, target_year: int = 2022, max_lookback: int = 5):
    """
    Try to get the value for feature in target_year.
    If missing, look back year-by-year up to max_lookback years.
    Returns (value, year) or (None, None).
    """
    for yr in range(target_year, target_year - max_lookback - 1, -1):
        sub = country_df[country_df["years"] == yr]
        if len(sub) and feature in sub.columns:
            v = sub[feature].iloc[0]
            if pd.notna(v):
                return float(v), yr
    return None, None

# Build AVAILABLE_FEATURES from arab_df (post filtering) and then ECDFs on that set
AVAILABLE_FEATURES: List[str] = [f for f in HIERARCHY.keys() if f in arab_df.columns]

# ════════════════════════════════════════
# TABS
# ════════════════════════════════════════


st.markdown("""
**Track the evolution of key vulnerability and resilience indicators** for Arab countries over time.
Use the **Pathway → Domain → Component → Driver** filters to focus on relevant parts of the framework.
""")

# Data normalization info box
with st.expander("ℹ️ **About Data Normalization**", expanded=False):
    col_norm1, col_norm2 = st.columns(2)
    with col_norm1:
        st.markdown("""
        **Currently showing: Global-normalized values (0-1 scale)**
        
        All indicators are normalized on a global scale from 0-1, where:
        - **1.0** = Best possible performance globally
        - **0.0** = Worst possible performance globally
        
        **🌍 Benefits of global normalization:**
                    
        ✅ **Cross-country prioritization** - Instantly spot which indicators are globally weak/strong  
        ✅ **Resource allocation** - "We're at 15th percentile for water" is persuasive and clear  
        ✅ **Portfolio balance** - Compare across domains (health, climate, institutions) without unit confusion  
        ✅ **Trade-off analysis** - See asymmetries clearly (e.g., mid-pack GDP but bottom-decile mortality)
        """)
    # with col_norm2:
    #     st.markdown("""
    #     **Coming soon: Raw values toggle**
        
    #     Raw values show actual units (%, $/capita, per 100k, etc.)
        
    #     **📊 Benefits of raw values:**
    #     ✅ **Operational targets** - Policies are funded and monitored in real units  
    #     ✅ **Impact modeling** - Analysts need raw values for elasticities (e.g., 1% unemployment drop saves X)  
    #     ✅ **Legal/SDG thresholds** - Many goals are absolute (<70 maternal deaths/100k; ≥90% water access)  
    #     ✅ **Direct interpretation** - Stakeholders understand "5% unemployment" vs "0.85 normalized score"
        
    #     *Option to toggle between views will be added in next release*
    #     """)
    # st.info("💡 **Best practice:** Use global-normalized for strategic prioritization, raw values for operational planning")

st.markdown("### 📊 Analytical Framework")
st.markdown("*Three-pathway risk assessment model: Vulnerability is the likelihood and structural exposure to shocks, and resilience is the policy-driven capacity to absorb the negative impact of shocks.*")

# Display the framework graphic
try:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("data/arm_hierarchy.png", width=700)
except:
    # Fallback if image doesn't exist
    st.info("""
    **Framework Overview:**
    
    🔴 **Conflict Pathway**
    - Conflict domain
    
    🟢 **Climate Pathway**  
    - Climate Hazards
    - Natural Resources
    
    🟠 **Development Pathway**
    - Economy
    - Society  
    - Institutions
    - Technology & Innovation
    
    Each domain assesses **Vulnerability** and **Resilience** to determine overall risk.
    """)

st.markdown("### 🌍 Select Country or Region")
available_countries = sorted(raw_df[raw_df['country_name'].isin(ARAB_COUNTRIES)]['country_name'].unique())
country_options = ["Arab Region"] + available_countries
selected_country = st.selectbox(
    "Select a country to analyze:",
    country_options,
    help="Choose an Arab country or the Arab Region to view development trajectories",
    key="country_trends_selector",
    label_visibility="collapsed"
)
    


# Helper function to categorize scores
def categorize_score(score):
    if pd.isna(score) or score is None:
        return None
    if score <= 0.20:
        return "Very Low"
    elif score <= 0.40:
        return "Low"
    elif score <= 0.60:
        return "Medium"
    elif score <= 0.80:
        return "High"
    else:
        return "Very High"

# Risk matrix lookup
def get_overall_risk(vulnerability_cat, resilience_cat):
    if vulnerability_cat is None or resilience_cat is None:
        return "No Data"
    
    risk_matrix = {
        ("Very High", "Very Low"): "Severe",
        ("Very High", "Low"): "Severe",
        ("Very High", "Medium"): "Significant",
        ("Very High", "High"): "Moderate",
        ("Very High", "Very High"): "Moderate",
        ("High", "Very Low"): "Severe",
        ("High", "Low"): "Significant",
        ("High", "Medium"): "Significant",
        ("High", "High"): "Moderate",
        ("High", "Very High"): "Moderate",
        ("Medium", "Very Low"): "Significant",
        ("Medium", "Low"): "Significant",
        ("Medium", "Medium"): "Moderate",
        ("Medium", "High"): "Minor",
        ("Medium", "Very High"): "Minor",
        ("Low", "Very Low"): "Moderate",
        ("Low", "Low"): "Moderate",
        ("Low", "Medium"): "Minor",
        ("Low", "High"): "Minor",
        ("Low", "Very High"): "Negligible",
        ("Very Low", "Very Low"): "Moderate",
        ("Very Low", "Low"): "Moderate",
        ("Very Low", "Medium"): "Minor",
        ("Very Low", "High"): "Negligible",
        ("Very Low", "Very High"): "Negligible",
    }
    return risk_matrix.get((vulnerability_cat, resilience_cat), "Unknown")

# Color mapping for overall risk
def get_risk_color(risk_level):
    color_map = {
        "Severe": "#e74c3c",
        "Significant": "#f39c12",
        "Moderate": "#f1c40f",
        "Minor": "#82e0aa",
        "Negligible": "#27ae60",
        "No Data": "#95a5a6"
    }
    return color_map.get(risk_level, "#ffffff")


# ════════════════════════════════════════════════════════════════
# OVERALL RISK ASSESSMENT TABLE
# ════════════════════════════════════════════════════════════════

# Build country (or regional average) dataset
if selected_country == "Arab Region":
    sub = raw_df[(raw_df['country_name'].isin(ARAB_COUNTRIES)) & (raw_df['years'] <= 2024)].copy()
    reg = sub.groupby('years').mean(numeric_only=True).reset_index()
    reg['country_name'] = "Arab Region"
    country_data = reg
else:
    country_data = raw_df[
        (raw_df['country_name'] == selected_country) &
        (raw_df['years'] <= 2024)
    ].copy()

# Data availability info
available_years = sorted(country_data['years'].unique()) if len(country_data) else []
if available_years:
    st.info(f"📅 Data available: {min(available_years)}-{max(available_years)}")
else:
    st.info("📅 No data available")


if len(country_data) > 0:
    st.markdown("---")
    st.subheader(f"🎯 Overall Risk Assessment: {selected_country}")
    st.markdown("*Based on latest available data across all domains*")
    
    # Build risk assessment by pathway and domain
    # Get most recent data (from 2020 onwards)
    recent_data = country_data[country_data['years'] >= 2020].copy()

    if len(recent_data) == 0:
        st.info("No data available from 2020 onwards to generate risk assessment")
        risk_assessment = []
    else:
        # Build risk assessment by pathway and domain
        risk_assessment = []
        
        # Define custom pathway order
        pathway_order = ['Conflict', 'Climate', 'Development']

        # Get all unique pathways and sort by custom order
        all_pathways = {HIERARCHY[f]['Pathway'] for f in AVAILABLE_FEATURES}
        ordered_pathways = [p for p in pathway_order if p in all_pathways]

        for pathway in ordered_pathways:
            pathway_domains = sorted({HIERARCHY[f]['Domain'] for f in AVAILABLE_FEATURES if HIERARCHY[f]['Pathway'] == pathway})                
            for domain in pathway_domains:
                # Get features for this pathway-domain combination (check against AVAILABLE_FEATURES, not country data)
                all_domain_features = [f for f in AVAILABLE_FEATURES 
                                if HIERARCHY[f]['Pathway'] == pathway 
                                and HIERARCHY[f]['Domain'] == domain]
                
                if not all_domain_features:
                    continue
                
                # Separate vulnerability and resilience features
                vuln_features = [f for f in all_domain_features if HIERARCHY[f]['Component'] == 'Vulnerability']
                res_features = [f for f in all_domain_features if HIERARCHY[f]['Component'] == 'Resilience']
                
                # Calculate averages using most recent non-null value for each feature
                vuln_values = []
                for f in vuln_features:
                    if f in recent_data.columns:  # Check if feature exists in this country's data
                        # Get most recent non-null value for this feature
                        feature_data = recent_data[['years', f]].dropna(subset=[f]).sort_values('years', ascending=False)
                        if len(feature_data) > 0:
                            val = feature_data[f].iloc[0]
                            if pd.notna(val):
                                # For vulnerability indicators that are "good when high", invert them
                                # because low values on these indicators = high vulnerability
                                if is_high_good(f):
                                    vuln_values.append(1.0 - float(val))
                                else:
                                    # For "bad when high" indicators, use value directly
                                    vuln_values.append(float(val))

                res_values = []
                for f in res_features:
                    if f in recent_data.columns:  # Check if feature exists in this country's data
                        # Get most recent non-null value for this feature
                        feature_data = recent_data[['years', f]].dropna(subset=[f]).sort_values('years', ascending=False)
                        if len(feature_data) > 0:
                            val = feature_data[f].iloc[0]
                            if pd.notna(val):
                                # For resilience indicators that are "bad when high", invert them
                                # because high values on these indicators = low resilience
                                if is_high_good(f):
                                    # For "good when high" indicators, use value directly
                                    res_values.append(float(val))
                                else:
                                    res_values.append(1.0 - float(val))
                
                # Calculate average scores
                vuln_avg = np.mean(vuln_values) if vuln_values else None
                res_avg = np.mean(res_values) if res_values else None
                
                # Categorize
                vuln_cat = categorize_score(vuln_avg)
                res_cat = categorize_score(res_avg)
                
                # Get overall risk
                overall_risk = get_overall_risk(vuln_cat, res_cat)
                
                # Always append the domain, even if there's no data
                risk_assessment.append({
                    'Pathway': pathway,
                    'Domain': domain,
                    'Vulnerability': vuln_cat if vuln_cat else "No Data",
                    'Resilience': res_cat if res_cat else "No Data",
                    'Overall Score': overall_risk,
                    '_vuln_avg': vuln_avg,
                    '_res_avg': res_avg
                })
    
    if risk_assessment:
        risk_df = pd.DataFrame(risk_assessment)
        
        # Style the dataframe
        def style_risk_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            
            for idx in df.index:
                risk = df.loc[idx, 'Overall Score']
                color = get_risk_color(risk)
                styles.loc[idx, 'Overall Score'] = f'background-color: {color}; color: white; font-weight: bold'
            
            return styles
        
        styled_risk = risk_df[['Pathway', 'Domain', 'Vulnerability', 'Resilience', 'Overall Score']].style.apply(
            style_risk_table, axis=None
        )
        
        st.dataframe(
            styled_risk.hide(axis='index'),
            use_container_width=True
        )
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            severe_count = sum(1 for r in risk_assessment if r['Overall Score'] == 'Severe')
            st.metric("🔴 Severe Risks", severe_count)
        with col2:
            sig_count = sum(1 for r in risk_assessment if r['Overall Score'] == 'Significant')
            st.metric("🟠 Significant Risks", sig_count)
        with col3:
            mod_count = sum(1 for r in risk_assessment if r['Overall Score'] == 'Moderate')
            st.metric("🟡 Moderate Risks", mod_count)
        with col4:
            minor_count = sum(1 for r in risk_assessment if r['Overall Score'] in ['Minor', 'Negligible'])
            st.metric("🟢 Minor/Negligible", minor_count)
        
        st.markdown("---")
    else:
        st.info("No data available to generate risk assessment for selected filters")



# Framework Filters
st.markdown("### 🎛️ Filter Indicators by Pathway or Domain")
pathways = sorted({HIERARCHY[f]['Pathway'] for f in AVAILABLE_FEATURES})

selected_pathway = st.selectbox("Choose a **Pathway** (Conflict, Climate, Development)", options=["All"] + pathways, index=0, key="sel_pathway")

# Helper function for cascade options (keep this local)
def cascade_options_local(current_filter: Dict[str, str]) -> Dict[str, List[str]]:
    meta_rows = [HIERARCHY[f] for f in AVAILABLE_FEATURES]
    df_meta = pd.DataFrame(meta_rows, index=AVAILABLE_FEATURES)
    mask = pd.Series(True, index=df_meta.index)
    if current_filter['Pathway'] != "All":
        mask &= (df_meta['Pathway'] == current_filter['Pathway'])
    doms = sorted(df_meta[mask]['Domain'].unique().tolist())
    if current_filter.get('Domain', "All") != "All":
        mask &= (df_meta['Domain'] == current_filter['Domain'])
    comps = sorted(df_meta[mask]['Component'].unique().tolist())
    if current_filter.get('Component', "All") != "All":
        mask &= (df_meta['Component'] == current_filter['Component'])
    drvs = sorted(df_meta[mask]['Driver'].unique().tolist())
    return {"Domains": doms, "Components": comps, "Drivers": drvs}

opts = cascade_options_local({"Pathway": selected_pathway, "Domain": "All", "Component": "All"})
domains = ["All"] + opts["Domains"]
selected_domain = st.selectbox("Choose a **Domain** (Conflict, Climate Hazards, Natural Resources, Economy, Institutions, Social, Technology & Innovation)", options=domains, index=0, key="sel_domain")


selected_component = "All"

selected_driver = "All"

# ════════════════════════════════════════════════════════════════
# Continue with existing trends table
# ════════════════════════════════════════════════════════════════

display_feature_cols = get_filtered_features(selected_pathway, selected_domain, selected_component, selected_driver)

# Prepare data for the trends table
if len(country_data) > 0 and len(display_feature_cols) > 0:
    # Build 2-year (closest) sampling
    min_year = min(available_years)
    max_year = max(available_years)
    display_years = [min_year]
    current_year = min_year + 2
    while current_year < max_year:
        if current_year in available_years:
            display_years.append(current_year)
        else:
            closest = min(available_years, key=lambda x: abs(x - current_year))
            if closest not in display_years:
                display_years.append(closest)
        current_year += 2
    if max_year not in display_years:
        display_years.append(max_year)
    display_years = sorted(list(set(display_years)))

    st.subheader(f"Key Drivers of Vulnerability and Resilience for {selected_country}")
    filter_tags = [f"Pathway: {selected_pathway}", f"Domain: {selected_domain}"]
    st.caption("Filters – " + " | ".join([t for t in filter_tags if not t.endswith("All")] or ["None (showing all in selection)"]))
    st.markdown(f"*Showing data for years: {', '.join(map(str, display_years))}*")

    # Build table rows
    trends_data = []
    for feat in display_feature_cols:
        row_data = {"Indicator": make_readable_name(feat)}
        values = []
        year_values = {}  # Store for later coloring
        
        for year in display_years:
            year_data = country_data[country_data['years'] == year]
            if len(year_data) > 0 and feat in year_data.columns:
                value = year_data[feat].iloc[0]
                value = (None if pd.isna(value) else float(np.round(value, 2)))
                values.append(value)
                year_values[str(year)] = value
                row_data[str(year)] = value
            else:
                values.append(None)
                year_values[str(year)] = None
                row_data[str(year)] = None

        valid_values = [v for v in values if v is not None and not pd.isna(v)]
        if len(valid_values) >= 2:
            trend = float(np.round(valid_values[-1] - valid_values[0], 3))
            high_good = is_high_good(feat)
            
            # Consider changes within ±0.03 as stable
            if abs(trend) <= 0.03:
                row_data["Trend"] = "➡️ Stable"
            else:
                if high_good:
                    row_data["Trend"] = "📈 Improving" if trend > 0 else "📉 Worsening"
                else:
                    row_data["Trend"] = "📈 Improving" if trend < 0 else "📉 Worsening"
        else:
            row_data["Trend"] = "⚠️ Insufficient data"

        meta = HIERARCHY.get(feat, {})
        row_data["Pathway"]   = meta.get("Pathway", "")
        row_data["Domain"]    = meta.get("Domain", "")
        row_data["Component"] = meta.get("Component", "")
        row_data["_feature_id"] = feat
        row_data["_year_values"] = year_values  # Store for styling

        trends_data.append(row_data)

    trends_df = pd.DataFrame(trends_data)

    first_cols = ["Pathway","Domain","Indicator"]
    year_cols = [str(y) for y in display_years]
    final_cols = first_cols + year_cols + ["Trend"]

    # ════════════════════════════════════════════════════════════════
    # ABSOLUTE QUARTILE COLORING (GLOBAL)
    # ════════════════════════════════════════════════════════════════
    def color_cell_global(feat_id: str, val: float) -> str:
        if val is None or pd.isna(val) or feat_id not in AVAILABLE_FEATURES:
            return 'background-color: #f0f0f0'
        
        # Use the value directly (it's already normalized 0-1)
        high_good = is_high_good(feat_id)
        
        if high_good:
            # For GOOD HIGH features: high values = dark green
            if val >= 0.75:
                return 'background-color: #1e8449; color: white'  # Dark green
            elif val >= 0.50:
                return 'background-color: #2ecc71; color: white'  # Green
            elif val >= 0.25:
                return 'background-color: #f39c12; color: white'  # Orange
            else:
                return 'background-color: #e74c3c; color: white'  # Red
        else:
            # For BAD HIGH features: high values = red
            if val >= 0.75:
                return 'background-color: #e74c3c; color: white'  # Red
            elif val >= 0.50:
                return 'background-color: #f39c12; color: white'  # Orange
            elif val >= 0.25:
                return 'background-color: #2ecc71; color: white'  # Green
            else:
                return 'background-color: #1e8449; color: white'  # Dark green

    def style_dataframe(df: pd.DataFrame, year_cols: list[str]) -> pd.io.formats.style.Styler:
        """
        - show only Pathway / Domain / Driver / Indicator + year cols + Trend
        - hide Component and _feature_id
        - hide index
        - still color year columns using the original df's _feature_id
        """

        # columns we don't want to display
        cols_to_drop = [c for c in ["Component", "_feature_id"] if c in df.columns]

        # 1) build the display dataframe first (this is what we will style)
        df_disp = df.drop(columns=cols_to_drop, errors="ignore").copy()

        # figure out which columns to keep, in what order
        base_cols = [c for c in ["Pathway", "Domain", "Driver", "Indicator"] if c in df_disp.columns]
        year_cols_in_df = [c for c in year_cols if c in df_disp.columns]
        tail_cols = [c for c in ["Trend"] if c in df_disp.columns]
        ordered_cols = base_cols + year_cols_in_df + tail_cols
        df_disp = df_disp.loc[:, ordered_cols]

        # keep feature ids from the original df for coloring
        feature_ids = df["_feature_id"].tolist() if "_feature_id" in df.columns else [None] * len(df_disp)

        # 2) build a styles DataFrame that matches df_disp (NOT the Styler!)
        styles = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)
        for i, feat_id in enumerate(feature_ids):
            if feat_id is None:
                continue
            for col in year_cols_in_df:
                val = df.loc[df.index[i], col] if col in df.columns else None
                styles.loc[df_disp.index[i], col] = color_cell_global(feat_id, val)

        # 3) create styler from the display df and apply styles
        styler = (
            df_disp.style
                .hide(axis="index")   # pandas 2.x syntax, cf. :contentReference[oaicite:2]{index=2}
                .apply(lambda _: styles, axis=None)  # apply our prebuilt style matrix
                .format({c: fmt2_trim for c in year_cols_in_df})
        )
        return styler

    # Prepare display dataframe (without helper columns)
    display_df = trends_df[final_cols].copy()

    # regenerate helper using our updated style function
    styled_df_fn = lambda df: style_dataframe(df, year_cols)

    # split by component (kept in the underlying df, not shown)
    vuln_df = trends_df[trends_df["Component"] == "Vulnerability"].copy()
    res_df  = trends_df[trends_df["Component"] == "Resilience"].copy()

    st.subheader(f"Key Drivers of Vulnerability for {selected_country}")
    if len(vuln_df):
        st.dataframe(
            style_dataframe(vuln_df, year_cols),
            use_container_width=True,
            height=min(600, 60 + 28 * len(vuln_df)),
        )
    else:
        st.info("No vulnerability indicators match the current filters.")

    st.subheader(f"Key Drivers of Resilience for {selected_country}")
    if len(res_df):
        st.dataframe(
            style_dataframe(res_df, year_cols),
            use_container_width=True,
            height=min(600, 60 + 28 * len(res_df)),
        )
    else:
        st.info("No resilience indicators match the current filters.")

    # Legend & summaries
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 🎨 Color Legend (Global Percentiles)")
        st.markdown("""
        Colors represent global performance after adjusting for indicator direction:
        - 🟩 **Dark Green** (≥ 75th percentile): Top quartile globally
        - 🟩 **Green** (50th-75th percentile): Above median performance
        - 🟠 **Orange** (25th-50th percentile): Below median performance
        - 🔴 **Red** (< 25th percentile): Bottom quartile globally
        - ⬜ **Gray**: No data available
        """)
    # with colB:
    #     st.markdown("### 🧭 Indicator Directionality")
    #     st.markdown("""
    #     **Higher is better (green when high):**
    #     GDP per capita, Water resources, Sanitation access, Health coverage, Government effectiveness, R&D expenditure, Internet usage
        
    #     **Higher is worse (red when high):**
    #     Mortality rates, Unemployment, Inequality (Gini), Debt ratios, Water stress, Displacement, Food insecurity, Aid dependence
        
    #     *The system automatically adjusts coloring based on each indicator's nature*
    #     """)

    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    improving = sum(1 for t in trends_df['Trend'] if 'Improving' in str(t))
    worsening = sum(1 for t in trends_df['Trend'] if 'Worsening' in str(t))
    stable = sum(1 for t in trends_df['Trend'] if 'Stable' in str(t))
    insufficient = sum(1 for t in trends_df['Trend'] if 'Insufficient' in str(t))
    
    with col1:
        st.metric("Improving Indicators", improving, help="Number of indicators showing improvement over time")
    with col2:
        st.metric("Worsening Indicators", worsening, help="Number of indicators showing deterioration")
    with col3:
        st.metric("Stable Indicators", stable, help="Number of indicators with minimal change")
    with col4:
        total_cells = len(display_feature_cols) * len(display_years)
        non_null_cells = sum(
            1 for _, row in trends_df.iterrows()
            for year in display_years
            if str(year) in row and pd.notna(row[str(year)])
        )
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        st.metric("Data Completeness", f"{completeness:.1f}%", help="Percentage of cells with available data")

    with st.expander("💪 View Strengths & Weaknesses"):
        st.markdown("Based on latest data")

        TARGET_YEAR = 2022
        rows_for_rank = []

        # we pool vulnerability + resilience indicators together from trends_df
        for _, row in trends_df.iterrows():
            feat_id = row["_feature_id"]
            indicator_label = row["Indicator"]

            # pull the 2022 value (or closest earlier) from the underlying country_data
            val_2022, actual_year = get_value_for_target_year(
                country_data,
                feat_id,
                target_year=TARGET_YEAR,
                max_lookback=5,   # look back to 2017 at most
            )

            if val_2022 is None:
                continue  # nothing to rank for this indicator

            hg = is_high_good(feat_id)

            # Strength logic:
            # - high-is-good  -> higher = stronger
            # - low-is-good   -> lower = stronger  (so 1 - value)
            if hg:
                strength_score = val_2022
            else:
                strength_score = 1.0 - val_2022

            rows_for_rank.append({
                "Indicator": indicator_label,
                "Feature ID": feat_id,
                "Value (2022)": float(val_2022),
                "Year Used": actual_year,
                "Is High Good": hg,
                "Strength Score": strength_score,
            })

        if not rows_for_rank:
            st.info("No data available for 2022 (or earlier) to compute strengths/weaknesses.")
        else:
            sw_df = pd.DataFrame(rows_for_rank)

            # ── Top 5 Strengths ───────────────────────────────────────────────
            # highest high-is-good values   → high Strength Score
            # lowest low-is-good values     → high Strength Score (because we did 1 - value)
            top_strengths = (
                sw_df.sort_values(["Strength Score", "Year Used"], ascending=[False, False])
                    .head(5)
            )

            # ── Top 5 Weaknesses ─────────────────────────────────────────────
            # highest low-is-good values    → lowest Strength Score (because 1 - value is small)
            # lowest high-is-good values    → lowest Strength Score
            top_weaknesses = (
                sw_df.sort_values(["Strength Score", "Year Used"], ascending=[True, False])
                    .head(5)
            )

            sL, sR = st.columns(2)

            with sL:
                st.markdown("*🌟 Top 5 Strengths*")
                st.caption("Strongest resilience and vulnerability indicators")
                for _, r in top_strengths.iterrows():
                    if r["Is High Good"]:
                        # e.g. 0.87 (higher is better)
                        st.markdown(
                            f"• *{r['Indicator']}* — {r['Value (2022)']:.2f}"
                            f"[{int(r['Year Used'])}]"
                        )
                    else:
                        # low is good → show that explicitly
                        st.markdown(
                            f"• *{r['Indicator']}* — {r['Value (2022)']:.2f}"
                            f"[{int(r['Year Used'])}]"
                        )

            with sR:
                st.markdown("*⚠️ Top 5 Weaknesses*")
                st.caption("Weakest resilience and vulnerability indicators")
                for _, r in top_weaknesses.iterrows():
                    if r["Is High Good"]:
                        # high-is-good but low value → weakness
                        st.markdown(
                            f"• *{r['Indicator']}* — {r['Value (2022)']:.2f}"
                            f"[{int(r['Year Used'])}]"
                        )
                    else:
                        # low-is-good but high value → weakness
                        st.markdown(
                            f"• *{r['Indicator']}* — {r['Value (2022)']:.2f}"
                            f"[{int(r['Year Used'])}]"
                        )

            # optional summary (kept from your original version)
            st.markdown("---")
            st.markdown("*📈 Performance Summary (2022 baseline)*")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Indicators scored", len(sw_df))
            with c2:
                st.metric("Top-quartile-ish (score ≥ 0.75)", int((sw_df["Strength Score"] >= 0.75).sum()))
            with c3:
                st.metric("Bottom (score < 0.25)", int((sw_df["Strength Score"] < 0.25).sum()))

    with st.expander("🔍 View Key Changes"):
        changes = []
        for _, row in trends_df.iterrows():
            values = [row[col] for col in year_cols if col in row.index]
            valid_values = [(i, v) for i, v in enumerate(values) if v is not None and not pd.isna(v)]
            if len(valid_values) >= 2:
                first_idx, first_val = valid_values[0]
                last_idx, last_val = valid_values[-1]
                
                # Calculate absolute change (not percentage)
                abs_change = float(np.round(last_val - first_val, 3))
                
                high_good = is_high_good(row["_feature_id"])
                
                # Determine if it's an improvement based on direction
                # For good high features: positive change = improvement
                # For bad high features: negative change = improvement
                is_improvement = (abs_change > 0) if high_good else (abs_change < 0)
                
                changes.append({
                    'Indicator': row['Indicator'],
                    'Absolute Change': abs_change,
                    'Direction': 'Improvement' if is_improvement else 'Deterioration',
                    'First Year': display_years[first_idx],
                    'Last Year': display_years[last_idx],
                    'First Value': first_val,
                    'Last Value': last_val
                })
        
        if changes:
            changes_df = pd.DataFrame(changes)
            cL, cR = st.columns(2)
            with cL:
                st.markdown("**🌟 Top 5 Improvements**")
                improvements = changes_df[changes_df['Direction'] == 'Improvement'].copy()
                # Sort by absolute magnitude of change (largest improvements)
                improvements['Magnitude'] = improvements['Absolute Change'].abs()
                improvements = improvements.nlargest(5, 'Magnitude', keep='first')
                if len(improvements) > 0:
                    for _, r in improvements.iterrows():
                        st.markdown(f"• **{r['Indicator']}**: +{r['Magnitude']:.3f} "
                                    f"({r['First Year']}-{r['Last Year']})")
                else:
                    st.info("No significant improvements found")
            with cR:
                st.markdown("**⚠️ Top 5 Deteriorations**")
                deteriorations = changes_df[changes_df['Direction'] == 'Deterioration'].copy()
                # Sort by absolute magnitude of change (largest deteriorations)
                deteriorations['Magnitude'] = deteriorations['Absolute Change'].abs()
                deteriorations = deteriorations.nlargest(5, 'Magnitude', keep='first')
                if len(deteriorations) > 0:
                    for _, r in deteriorations.iterrows():
                        st.markdown(f"• **{r['Indicator']}**: -{r['Magnitude']:.3f} "
                                    f"({r['First Year']}-{r['Last Year']})")
                else:
                    st.info("No significant deteriorations found")
            


# Footer
st.markdown("---")
st.caption("Arab Risk Monitor v2.0 | Data sources: World Bank, UN, V-Dem, and other international organizations")
st.caption("Note: All values are globally normalized (0-1 scale) for cross-country comparison.")