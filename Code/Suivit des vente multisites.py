import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Sales Expert Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- 2. FONCTIONS DE CHARGEMENT ET CACHE ---

@st.cache_data
def load_data():
    """Charge les données et prépare les colonnes temporelles."""
    # Remplacez par votre chemin de fichier
    path = 'Data/Online Sales Data.csv'
    df = pd.read_csv(path)
    
    # Conversion date
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    
    # Traduction des mois en Français
    mois_fr = {
        1:'Janv', 2:'Févr', 3:'Mars', 4:'Avril', 5:'Mai', 6:'Juin', 
        7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'
    }
    
    df['month_num'] = df['Date'].dt.month
    df['month'] = df['month_num'].map(mois_fr)
    df['year'] = df['Date'].dt.year
    
    return df

@st.cache_resource
def train_prophet_model(data):
    # 1. Préparation des données
    df_p = data.groupby('Date')['Total Revenue'].sum().reset_index()
    df_p.columns = ['ds', 'y']
    
    # 2. Ajout des bornes (Crucial pour éviter le négatif)
    # Le 'cap' est une estimation haute (ex: 20% de plus que le max historique)
    df_p['cap'] = df_p['y'].max() * 1.5 
    df_p['floor'] = 0
    
    # 3. Initialisation avec croissance logistique
    
    m = Prophet(
        growth='logistic', 
        yearly_seasonality=True, 
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    m.fit(df_p)
    
    # 4. Prévision avec les mêmes bornes
    future = m.make_future_dataframe(periods=30, freq='D')
    future['cap'] = df_p['y'].max() * 1.5
    future['floor'] = 0
    
    forecast = m.predict(future)
    
    # Sécurité ultime : on force mathématiquement les valeurs négatives à 0
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    
    return forecast

# --- 3. RÉCUPÉRATION DES DONNÉES ---
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")
    st.stop()

# --- 4. BARRE LATÉRALE (FILTRES) ---
st.sidebar.header('⚙️ Configuration')

# Filtre par Année
years = sorted(df['year'].unique())
selected_year = st.sidebar.multiselect('Choisir les années', years, default=years)

# Filtre par Région
regions = df['Region'].unique()
selected_region = st.sidebar.multiselect('Choisir les régions', regions, default=regions)

# Application des filtres au DF pour les onglets 1 et 2
df_filtered = df[
    (df['year'].isin(selected_year)) & 
    (df['Region'].isin(selected_region))
]

# --- 5. INTERFACE PRINCIPALE ---
st.title('📊 Dashboard Suivi des Ventes & IA')

# Métriques globales en haut
c1, c2, c3 = st.columns(3)
total_rev = df_filtered['Total Revenue'].sum()
total_units = df_filtered['Units Sold'].sum()
c1.metric("Chiffre d'Affaires Total", f"{total_rev:,.2f} €")
c2.metric("Unités Vendues", f"{total_units:,}")
c3.metric("Nombre de Transactions", len(df_filtered))

st.write("---")

tab1, tab2, tab3 = st.tabs(['📈 Analyses Visuelles', '📋 Registre de Données', '🔮 Prévisions IA'])

# --- ONGLET 1 : GRAPHIQUES ---
with tab1:
    col_left, col_right = st.columns(2)
    
    # Ordre des mois pour le tri Plotly
    ordre_mois = ['Janv', 'Févr', 'Mars', 'Avril', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc']
    
    with col_left:
        # CA par Mois
        df_month = df_filtered.groupby('month')['Total Revenue'].sum().reset_index()
        fig_bar = px.bar(
            df_month, x='month', y='Total Revenue',
            category_orders={"month": ordre_mois},
            title="Répartition Mensuelle du CA",
            color_discrete_sequence=['#00CC96'],
            template='simple_white',
            text_auto='.2s'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        # CA par Catégorie de Produit
        df_cat = df_filtered.groupby('Product Category')['Total Revenue'].sum().reset_index()
        fig_pie = px.pie(
            df_cat, names='Product Category', values='Total Revenue',
            title="CA par Catégorie",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- ONGLET 2 : DATA ---
with tab2:
    st.dataframe(df_filtered, use_container_width=True)

# --- ONGLET 3 : PRÉVISIONS (PROPHET) ---
with tab3:
    st.header("Estimation du Chiffre d'Affaires futur")
    st.info("Le modèle analyse l'historique global pour prédire les 30 prochains jours.")
    
    with st.spinner('L\'IA analyse les tendances...'):
        # On entraîne sur le dataset complet pour avoir plus de contexte
        forecast = train_prophet_model(df)
        
        # Graphique de prédiction
        fig_forecast = px.line(
            forecast, x='ds', y='yhat',
            title="Prédiction du CA (30 jours)",
            labels={'ds': 'Date', 'yhat': 'CA Estimé (€)'},
            template='plotly_white'
        )
        # Ajout de l'aire d'incertitude
        fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False)
        fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(0,176,246,0.2)', name='Incertitude')
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Métriques de prévision
        next_month_val = forecast.tail(30)['yhat'].sum()
        st.success(f"**Prévision pour le mois prochain : {next_month_val:,.2f} €**")