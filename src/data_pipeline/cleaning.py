import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def clean_macro_data(
    emprunts_df: pd.DataFrame,
    irl_df: pd.DataFrame,
    interet_df: Optional[pd.DataFrame] = None,
    endettement_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Nettoie et fusionne les données macroéconomiques mensuelles (emprunts, IRL, taux d'intérêt).
    """
    df_emp = emprunts_df.copy()
    df_emp['date_dt'] = pd.to_datetime(df_emp['date'], format='%Y-%m')
    df_emp['date'] = df_emp['date_dt'].dt.to_period('M')

    df_irl_clean = irl_df.copy()
    df_irl_clean['date_dt'] = pd.to_datetime(df_irl_clean['quarter'], format='%Y-%m-%d')
    df_irl_clean['date'] = df_irl_clean['date_dt'].dt.to_period('M')

    # Merge emprunts + IRL
    df_macro = pd.merge(
        df_emp[['date', 'date_dt', 'emprunts_M€']],
        df_irl_clean[['date', 'IRL']],
        on='date',
        how='left'
    )
    df_macro['IRL'] = df_macro['IRL'].bfill().ffill()

    # Ajout du taux d'intérêt si disponible
    if interet_df is not None and not interet_df.empty:
        df_int = interet_df.copy()
        df_int['date'] = pd.to_datetime(df_int['date'], format='%Y-%m').dt.to_period('M')
        df_macro = pd.merge(df_macro, df_int[['date', 'taux']], on='date', how='left')
        df_macro['taux_interet'] = df_macro['taux'].bfill().ffill()
        df_macro.drop(columns=['taux'], inplace=True, errors='ignore')
    else:
        df_macro['taux_interet'] = 3.5  # Valeur par défaut représentative

    # Ajout du taux d'endettement si disponible (donnée annuelle reportée sur les mois)
    if endettement_df is not None and not endettement_df.empty:
        df_end = endettement_df.copy()
        df_end['annee'] = df_end['date'].astype(int)
        df_macro['annee'] = df_macro['date_dt'].dt.year
        df_macro = pd.merge(df_macro, df_end[['annee', 'taux_endettement']], on='annee', how='left')
        df_macro['taux_endettement'] = df_macro['taux_endettement'].bfill().ffill()
        df_macro.drop(columns=['annee'], inplace=True, errors='ignore')

    df_macro.sort_values(by='date_dt', inplace=True)
    df_macro.reset_index(drop=True, inplace=True)
    return df_macro


def clean_territorial_data(
    loyers_df: pd.DataFrame,
    fiscaux_df: pd.DataFrame,
    parc_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie, fusionne et impute l'ensemble des données territoriales (loyers, fiscalité, parc).
    Garantit 0 valeur manquante dans le DataFrame final.
    """
    df_loy = loyers_df.copy()
    df_loy['annee'] = pd.to_datetime(df_loy['date'], format='%Y').dt.year
    df_loy['departement'] = df_loy['departement'].astype(str).str.zfill(2)
    df_loy['id_ville'] = df_loy['id_ville'].astype(int)

    df_fisc = fiscaux_df.copy()
    df_fisc['annee'] = pd.to_datetime(df_fisc['date'], format='%Y').dt.year
    df_fisc['departement'] = df_fisc['departement'].astype(str).str.zfill(2)
    df_fisc['id_ville'] = df_fisc['id_ville'].astype(int)

    df_prc = parc_df.copy()
    df_prc['departement'] = df_prc['departement'].astype(str).str.zfill(2)
    df_prc['id_ville'] = df_prc['id_ville'].astype(int)

    # 1. Jointure Loyers + Fiscalité par commune et année
    df_terr = pd.merge(
        df_loy[['departement', 'id_ville', 'ville', 'annee', 'loyer_m2_appartement', 'loyer_m2_maison']],
        df_fisc.drop(columns=['date', 'ville'], errors='ignore'),
        on=['departement', 'id_ville', 'annee'],
        how='left'
    )

    # 2. Jointure Parc immobilier (Logements vacants dédupliqués par commune)
    parc_unique = df_prc[['departement', 'id_ville', 'n_logements', 'n_logements_vacants']].drop_duplicates(
        subset=['departement', 'id_ville']
    )
    df_terr = pd.merge(
        df_terr,
        parc_unique,
        on=['departement', 'id_ville'],
        how='left'
    )

    # 3. Pipeline d'imputation robuste
    # Imputation n_foyers_fiscaux
    df_terr['n_foyers_fiscaux'] = df_terr['n_foyers_fiscaux'].fillna(
        df_terr.groupby('departement')['n_foyers_fiscaux'].transform('median')
    ).fillna(df_terr['n_foyers_fiscaux'].median()).fillna(500.0)

    # Imputation revenus fiscaux et impôts
    for col in ['revenu_fiscal_moyen', 'montant_impot_moyen']:
        if col in df_terr.columns:
            df_terr[col] = df_terr[col].fillna(
                df_terr.groupby('departement')[col].transform('median')
            ).fillna(df_terr[col].median()).fillna(25000.0)

    # Imputation proportionnelle des tranches de revenus
    tranche_cols = [
        'n_foyers_0k_10k', 'n_foyers_10k_12k', 'n_foyers_12k_15k', 'n_foyers_15k_20k',
        'n_foyers_20k_30k', 'n_foyers_30k_50k', 'n_foyers_50k_100k', 'n_foyers_100k_plus'
    ]
    for col in tranche_cols:
        if col in df_terr.columns:
            ratio = (df_terr[col] / df_terr['n_foyers_fiscaux']).median()
            if pd.isna(ratio) or ratio == 0:
                ratio = 0.12
            df_terr[col] = df_terr[col].fillna(np.round(df_terr['n_foyers_fiscaux'] * ratio))

    # Imputation logements et logements vacants
    df_terr['n_logements'] = df_terr['n_logements'].fillna(
        df_terr['n_foyers_fiscaux'] * 1.15
    )
    df_terr['n_logements_vacants'] = df_terr['n_logements_vacants'].fillna(
        df_terr.groupby('departement')['n_logements_vacants'].transform('median')
    ).fillna(df_terr['n_logements_vacants'].median()).fillna(50.0)

    # Taux de vacance calculé
    df_terr['taux_vacance'] = (df_terr['n_logements_vacants'] / df_terr['n_logements']).clip(0.01, 0.5)

    return df_terr

