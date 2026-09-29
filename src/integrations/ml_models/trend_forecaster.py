import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.integrations.data_pipeline.feature_store import FeatureStore


class TrendForecaster:
    """
    Modèle de simulation et de projection temporelle de rentabilité à horizon 1 à 5 ans.
    Prend en compte l'inflation (IRL), les taux d'emprunt, les charges et la croissance historique.
    """

    def __init__(self):
        self.feature_store = FeatureStore.get_instance()

    def forecast_profitability(
        self,
        prix_achat: float,
        loyer_mensuel: Optional[float] = None,
        ville: Optional[str] = None,
        departement: Optional[str] = None,
        horizon_annees: int = 2,
        charges_annuelles_pct: float = 1.2,
        taxe_fonciere_pct: float = 1.0,
        taux_croissance_prix_annuel: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calcule la rentabilité prévisionnelle détaillée d'un bien immobilier.
        """
        # 1. Recherche du taux de croissance local ou par défaut
        if taux_croissance_prix_annuel is None:
            if ville:
                profile = self.feature_store.get_city_profile(ville, departement)
                if profile is not None and profile.get('revenu_fiscal_moyen', 0) > 35000:
                    taux_croissance_prix_annuel = 0.035  # Zone dynamique
                else:
                    taux_croissance_prix_annuel = 0.022  # Zone standard
            else:
                taux_croissance_prix_annuel = 0.025

        # 2. Si loyer non fourni, estimation selon le rendement moyen
        if loyer_mensuel is None or loyer_mensuel <= 0:
            rendement_default = 0.058
            loyer_mensuel = (prix_achat * rendement_default) / 12

        # 3. Évolution trimestrielle / annuelle sur la période
        months = horizon_annees * 12
        timeline_data: List[Dict[str, Any]] = []

        current_valeur = prix_achat
        total_loyers_encaisses = 0.0
        total_charges = 0.0

        growth_monthly = (1 + taux_croissance_prix_annuel) ** (1 / 12) - 1
        irl_growth_monthly = (1 + 0.025) ** (1 / 12) - 1  # Croissance moyenne de l'IRL à 2.5%/an
        current_loyer = loyer_mensuel

        charges_mensuelles = (prix_achat * (charges_annuelles_pct + taxe_fonciere_pct) / 100) / 12

        for m in range(1, months + 1):
            current_valeur *= (1 + growth_monthly)
            current_loyer *= (1 + irl_growth_monthly)
            total_loyers_encaisses += current_loyer
            total_charges += charges_mensuelles

            if m % 6 == 0 or m == months:
                timeline_data.append({
                    "mois": m,
                    "valeur_bien": round(current_valeur, 2),
                    "loyers_cumules": round(total_loyers_encaisses, 2),
                    "charges_cumulees": round(total_charges, 2),
                    "gain_total_net": round((current_valeur - prix_achat) + (total_loyers_encaisses - total_charges), 2)
                })

        plus_value = current_valeur - prix_achat
        revenus_locatifs_nets = total_loyers_encaisses - total_charges
        gain_total = plus_value + revenus_locatifs_nets
        roi_global_pct = (gain_total / prix_achat) * 100 if prix_achat > 0 else 0.0
        rendement_net_annuel_pct = (revenus_locatifs_nets / horizon_annees / prix_achat) * 100 if prix_achat > 0 else 0.0

        est_rentable = gain_total > 0 and roi_global_pct >= (horizon_annees * 3.0)

        # Verdict textuel
        if roi_global_pct >= (horizon_annees * 5.0):
            avis = "🚀 Projet Très Rentable"
            details = "Forte création de valeur combinant plus-value et excellent rendement locatif net."
        elif roi_global_pct >= (horizon_annees * 2.5):
            avis = "✅ Projet Rentable et Équilibré"
            details = "Rendement conforme aux standards de marché avec valorisation régulière du capital."
        else:
            avis = "⚠️ Rentabilité Faible ou Risquée"
            details = "Les charges et la faible croissance locale limitent le rendement net."

        return {
            "prix_achat_initial": round(prix_achat, 2),
            "loyer_mensuel_initial": round(loyer_mensuel, 2),
            "horizon_annees": horizon_annees,
            "valeur_future_estimee": round(current_valeur, 2),
            "plus_value_estimee": round(plus_value, 2),
            "total_loyers_bruts": round(total_loyers_encaisses, 2),
            "total_charges_et_taxes": round(total_charges, 2),
            "revenus_locatifs_nets": round(revenus_locatifs_nets, 2),
            "gain_total_net": round(gain_total, 2),
            "roi_global_pct": round(roi_global_pct, 2),
            "rendement_net_annuel_pct": round(rendement_net_annuel_pct, 2),
            "est_rentable": est_rentable,
            "verdict": avis,
            "explication": details,
            "timeline": timeline_data
        }

