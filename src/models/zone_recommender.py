import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.data_pipeline.feature_store import FeatureStore


class ZoneRecommender:
    """
    Moteur de recommandation multicritères pour trouver les meilleures zones d'achat
    ou de location selon le budget, les ressources fiscales du foyer et les objectifs.
    """

    def __init__(self):
        self.feature_store = FeatureStore.get_instance()

    def evaluate_city_opportunity(
        self,
        budget: float,
        action: str,  # "achat" ou "location"
        ville: str,
        type_bien: str = "appartement",
        surface_m2: float = 60.0,
        revenu_foyer_annuel: Optional[float] = None,
        departement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évalue si un projet d'achat ou de location est une bonne idée dans une ville précise (Cas d'utilisation 1).
        """
        action_clean = action.strip().lower()
        is_achat = "ach" in action_clean or "buy" in action_clean
        is_apt = "app" in type_bien.lower()
        
        city = self.feature_store.get_city_profile(ville, departement)
        
        if city is None:
            return {
                "ville": ville,
                "recommandable": False,
                "score": 50.0,
                "avis": f"Ville '{ville}' non trouvée dans le référentiel territorial.",
                "recommandation": "Veuillez vérifier l'orthographe de la ville ou préciser le département."
            }

        loyer_m2 = city['loyer_m2_appartement'] if is_apt else city['loyer_m2_maison']
        prix_m2 = city['prix_m2_appartement'] if is_apt else city['prix_m2_maison']
        revenu_local = city['revenu_fiscal_moyen']
        taux_vacance = city['taux_vacance']

        loyer_mensuel_estime = loyer_m2 * surface_m2
        prix_achat_estime = prix_m2 * surface_m2

        # Taux d'effort calculé selon le revenu fourni ou le revenu fiscal moyen local
        revenu_ref = revenu_foyer_annuel if revenu_foyer_annuel and revenu_foyer_annuel > 0 else revenu_local
        revenu_mensuel_ref = revenu_ref / 12

        if is_achat:
            # Évaluation Achat
            budget_mensuel_credit = (budget * 0.0055) if budget > 10000 else (prix_achat_estime * 0.0055) # Taux mensualité approximatif
            taux_effort = (budget_mensuel_credit / revenu_mensuel_ref) * 100
            faisabilite_budget = budget >= prix_achat_estime if budget > 10000 else budget_mensuel_credit <= (revenu_mensuel_ref * 0.35)

            score = 100 - min(max(taux_effort - 25, 0) * 2.5, 60) + (city['rendement_brut_appartement'] * 2.5) - (taux_vacance * 50)
            score = round(float(np.clip(score, 10, 99)), 1)

            bonne_idee = faisabilite_budget and score >= 60 and taux_effort <= 35

            if bonne_idee:
                verdict = "✅ Excellente Opportunité d'Achat"
                conseil = f"Le prix moyen estimé ({prix_achat_estime:,.0f} €) est adapté à votre budget. Le marché local présente un bon dynamisme et une vacance maîtrisée ({taux_vacance*100:.1f}%)."
            elif faisabilite_budget:
                verdict = "⚖️ Faisable mais sous Conditions"
                conseil = f"Le budget permet l'acquisition ({prix_achat_estime:,.0f} €), mais surveillez le rendement net et les charges locales."
            else:
                verdict = "⚠️ Risque Financier / Budget Juste"
                conseil = f"Le prix moyen ({prix_achat_estime:,.0f} €) dépasse la capacité recommandée (taux d'effort estimé à {taux_effort:.1f}% > 35%)."

        else:
            # Évaluation Location
            taux_effort = (loyer_mensuel_estime / revenu_mensuel_ref) * 100
            bonne_idee = taux_effort <= 33.0
            score = round(float(np.clip(100 - (taux_effort * 1.8), 15, 95)), 1)

            if bonne_idee:
                verdict = "✅ Excellente Option de Location"
                conseil = f"Loyer estimé à {loyer_mensuel_estime:.0f} €/mois, représentant un taux d'effort sain de {taux_effort:.1f}% de vos revenus."
            else:
                verdict = "⚠️ Loyer Élevé pour ce Profil de Revenus"
                conseil = f"Loyer estimé à {loyer_mensuel_estime:.0f} €/mois. Le taux d'effort de {taux_effort:.1f}% dépasse le seuil de solvabilité usuel (33%)."

        return {
            "ville": city['ville'],
            "departement": city['departement'],
            "action": "Achat" if is_achat else "Location",
            "type_bien": "Appartement" if is_apt else "Maison",
            "surface_m2": surface_m2,
            "prix_achat_estime": round(prix_achat_estime, 2),
            "loyer_mensuel_estime": round(loyer_mensuel_estime, 2),
            "taux_effort_pct": round(taux_effort, 1),
            "score_opportunite": score,
            "bonne_idee": bonne_idee,
            "verdict": verdict,
            "conseil": conseil,
            "revenu_moyen_commune": round(revenu_local, 2),
            "taux_vacance_pct": round(taux_vacance * 100, 2)
        }

    def recommend_top_zones(
        self,
        action: str = "achat",
        budget: Optional[float] = None,
        revenu_foyer_annuel: Optional[float] = None,
        type_bien: str = "appartement",
        departement_pref: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Trouve le top des communes idéales pour acheter ou louer selon les ressources du foyer (Cas d'utilisation 3).
        """
        df = self.feature_store.df_features.copy()
        is_achat = "ach" in action.lower() or "buy" in action.lower()
        is_apt = "app" in type_bien.lower()

        # Filtrage départemental si spécifié
        if departement_pref:
            dept_code = str(departement_pref).zfill(2)
            df = df[df['departement'] == dept_code]

        # Calcul des prix selon le type de bien
        surface_std = 70.0 if not is_apt else 50.0
        prix_col = 'prix_m2_appartement' if is_apt else 'prix_m2_maison'
        loyer_col = 'loyer_m2_appartement' if is_apt else 'loyer_m2_maison'

        df['prix_total'] = df[prix_col] * surface_std
        df['loyer_total'] = df[loyer_col] * surface_std

        # Filtre de budget maximum si fourni
        if budget and budget > 0:
            if is_achat:
                df = df[df['prix_total'] <= budget * 1.15]
            else:
                df = df[df['loyer_total'] <= (budget if budget < 5000 else budget / 12)]

        # Filtre / Scoring selon le revenu du foyer
        if revenu_foyer_annuel and revenu_foyer_annuel > 0:
            # Recherche de communes avec un tissu socio-économique proche ou accessible
            diff_revenu = np.abs(df['revenu_fiscal_moyen'] - revenu_foyer_annuel)
            df['score_adhesion_revenu'] = (1 - (diff_revenu / df['revenu_fiscal_moyen'].max())) * 30
        else:
            df['score_adhesion_revenu'] = 20.0

        if is_achat:
            # Classement Achat / Investissement
            df['score_final'] = (
                df['score_attractivite_achat'] * 0.5 +
                (df['rendement_brut_appartement'] * 4.0) +
                df['score_adhesion_revenu']
            )
        else:
            # Classement Location
            df['score_final'] = (
                df['score_attractivite_location'] * 0.6 +
                (1 - df['taux_vacance']) * 20 +
                df['score_adhesion_revenu']
            )

        top_df = df.sort_values(by='score_final', ascending=False).drop_duplicates(subset=['departement', 'id_ville']).head(top_k)

        results = []
        for _, row in top_df.iterrows():
            results.append({
                "ville": row['ville'],
                "departement": row['departement'],
                "loyer_m2": round(row[loyer_col], 2),
                "prix_m2": round(row[prix_col], 2),
                "loyer_mensuel_moyen": round(row['loyer_total'], 0),
                "prix_achat_moyen": round(row['prix_total'], 0),
                "rendement_brut_pct": round(row['rendement_brut_appartement'], 2),
                "revenu_fiscal_moyen": round(row['revenu_fiscal_moyen'], 0),
                "taux_vacance_pct": round(row['taux_vacance'] * 100, 1),
                "score_recommandation": round(float(row['score_final']), 1)
            })

        return results

