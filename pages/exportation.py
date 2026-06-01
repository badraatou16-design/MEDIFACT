import streamlit as st
import pandas as pd
import numpy as np
import io

def show():
    st.title("Exportation des Résultats")

    if st.session_state.df is None:
        st.warning("Aucun dataset chargé.")
        return

    df = st.session_state.df
    orig = st.session_state.df_original

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes dans le dataset nettoyé", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    if orig is not None:
        c3.metric("Lignes supprimées", orig.shape[0] - df.shape[0])

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Données", "Statistiques", "Rapport"])

    with tab1:
        st.subheader("Exporter le Dataset Nettoyé")

        # CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Télécharger en CSV",
                           csv_buf.getvalue(), "dataset_cleaned.csv", "text/csv")

        # Excel
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Données nettoyées")
            if orig is not None:
                orig.to_excel(writer, index=False, sheet_name="Données originales")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                df[num_cols].describe().round(3).to_excel(writer, sheet_name="Statistiques")
        st.download_button("Télécharger en Excel",
                           excel_buf.getvalue(), "dataset_cleaned.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab2:
        st.subheader("Exporter les Statistiques Descriptives")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            stats = df[num_cols].describe().round(4)
            st.dataframe(stats, use_container_width=True)
            csv_stats = stats.to_csv()
            st.download_button("Télécharger les statistiques (CSV)",
                               csv_stats, "statistiques.csv", "text/csv")

    with tab3:
        st.subheader("Rapport d'Analyse Automatique")
        report = _generate_report(df, orig)
        st.text_area("Rapport:", report, height=400)
        st.download_button("Télécharger le rapport (TXT)",
                           report, "rapport_analyse.txt", "text/plain")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Historique des Traitements")
        for i, action in enumerate(st.session_state.history, 1):
            st.markdown(f"{i}. {action}")
        hist_text = "\n".join([f"{i}. {a}" for i, a in enumerate(st.session_state.history, 1)])
        st.download_button("Télécharger l'historique",
                           hist_text, "historique.txt", "text/plain")

def _generate_report(df, orig):
    lines = [
        "=" * 60,
        "RAPPORT D'ANALYSE — BIOMED DATAHUB",
        "Healthcare Dataset Collection (Sujet 20)",
        "=" * 60,
        "",
        "1. INFORMATIONS GÉNÉRALES",
        f"Lignes (nettoyé):  {df.shape[0]}",
        f"Colonnes:          {df.shape[1]}",
    ]
    if orig is not None:
        lines.append(f"Lignes (original): {orig.shape[0]}")
        lines.append(f"Lignes supprimées: {orig.shape[0] - df.shape[0]}")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    lines += ["",
              "2. TYPES DE COLONNES",
              f"Numériques ({len(num_cols)}): {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}",
              f"Catégorielles ({len(cat_cols)}): {', '.join(cat_cols[:5])}{'...' if len(cat_cols) > 5 else ''}",
              "",
              "3. QUALITÉ DES DONNÉES",
              f"Valeurs manquantes: {df.isnull().sum().sum()}",
              f"Doublons: {df.duplicated().sum()}",
              "",
              "4. STATISTIQUES DESCRIPTIVES"]
    for c in num_cols[:8]:
        s = df[c].dropna()
        lines.append(f" {c}: mean={s.mean():.3f}, std={s.std():.3f}, min={s.min():.3f}, max={s.max():.3f}")

    lines += ["",
              "5. HISTORIQUE DES TRAITEMENTS"]
    for i, action in enumerate(st.session_state.history, 1):
        lines.append(f" {i}. {action}")
    lines += ["", "=" * 60]
    return "\n".join(lines)
