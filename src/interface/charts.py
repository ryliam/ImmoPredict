import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any


def create_forecast_timeline_chart(timeline: List[Dict[str, Any]]) -> go.Figure:
    """
    Génère un graphique interactif montrant l'évolution de la valeur du bien
    et le gain net cumulé au fil des mois (Cas 2).
    """
    if not timeline:
        return go.Figure()

    df = pd.DataFrame(timeline)
    
    fig = go.Figure()

    # Courbe Valeur du bien
    fig.add_trace(go.Scatter(
        x=[f"Mois {m}" for m in df['mois']],
        y=df['valeur_bien'],
        mode='lines+markers',
        name='Valeur estimée du bien (€)',
        line=dict(color='#2563EB', width=3),
        marker=dict(size=8)
    ))

    # Courbe Gain Net Cumulé
    fig.add_trace(go.Bar(
        x=[f"Mois {m}" for m in df['mois']],
        y=df['gain_total_net'],
        name='Gain net cumulé (€)',
        marker_color='#10B981',
        opacity=0.7
    ))

    fig.update_layout(
        title="📈 Projection de Valorisation et Gain Net (Horizon Prévisionnel)",
        xaxis_title="Période",
        yaxis_title="Montant (€)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        height=380
    )
    return fig


def create_top_zones_chart(zones: List[Dict[str, Any]]) -> go.Figure:
    """
    Génère un graphique à barres horizontales comparant les meilleures communes (Cas 3).
    """
    if not zones:
        return go.Figure()

    df = pd.DataFrame(zones)
    
    fig = px.bar(
        df,
        x='score_recommandation',
        y='ville',
        orientation='h',
        color='rendement_brut_pct',
        color_continuous_scale='Viridis',
        text='score_recommandation',
        labels={
            'score_recommandation': 'Score Global (/100)',
            'ville': 'Commune',
            'rendement_brut_pct': 'Rendement Brut (%)'
        },
        title="🏆 Classement des Communes Recommandées selon votre Profil"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    return fig


def create_macro_market_chart(chart_data: Dict[str, Any]) -> go.Figure:
    """
    Génère un graphique des indicateurs macroéconomiques (Taux d'intérêt et IRL).
    """
    macro = chart_data.get("macro", {})
    dates = macro.get("dates", [])
    irl = macro.get("irl", [])
    taux = macro.get("taux_interet", [])

    if not dates:
        return go.Figure()

    fig = go.Figure()

    # Série IRL
    fig.add_trace(go.Scatter(
        x=dates,
        y=irl,
        name="Indice IRL (Loyers)",
        line=dict(color="#3B82F6", width=2.5)
    ))

    # Série Taux d'intérêt
    fig.add_trace(go.Scatter(
        x=dates,
        y=taux,
        name="Taux d'Intérêt (%)",
        line=dict(color="#EF4444", width=2, dash='dot'),
        yaxis="y2"
    ))

    fig.update_layout(
        title="📊 Conjoncture Immobilière Récente (IRL vs Taux de Crédit)",
        xaxis_title="Date",
        yaxis=dict(title="Indice IRL (base 100)"),
        yaxis2=dict(
            title="Taux d'intérêt (%)",
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        height=360
    )
    return fig

