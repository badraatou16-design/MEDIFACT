import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io

def show():
    st.title("Analyse Exploratoire des Données")

    if st.session_state.df is None:
        st.warning("Aucun dataset chargé. Veuillez d'abord importer vos données.")
        return

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Statistiques", "Corrélations", "Distributions",
        "Valeurs manquantes", "Doublons"
    ])

    #  TAB 1: Statistics 
    with tab1:
        st.subheader("Statistiques Descriptives")
        if not num_cols:
            st.info("Aucune colonne numérique.")
        else:
            col = st.selectbox("Choisir une colonne:", num_cols, key="stat_col")
            data = df[col].dropna()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Moyenne", f"{data.mean():.3f}")
            c2.metric("Médiane", f"{data.median():.3f}")
            c3.metric("Mode", f"{data.mode().iloc[0]:.3f}" if len(data.mode()) > 0 else "N/A")
            c4.metric("Écart-type", f"{data.std():.3f}")
            c1b, c2b, c3b, c4b = st.columns(4)
            c1b.metric("Variance", f"{data.var():.3f}")
            c2b.metric("Min", f"{data.min():.3f}")
            c3b.metric("Max", f"{data.max():.3f}")
            c4b.metric("Q1–Q3", f"{data.quantile(0.25):.2f} – {data.quantile(0.75):.2f}")

            st.markdown("---")
            st.subheader("Tableau Récapitulatif Complet")
            stats_data = []
            for c in num_cols:
                s = df[c].dropna()
                stats_data.append({
                    "Colonne": c, "Moyenne": round(s.mean(), 3), "Médiane": round(s.median(), 3),
                    "Écart-type": round(s.std(), 3), "Min": round(s.min(), 3), "Max": round(s.max(), 3),
                    "Q25": round(s.quantile(0.25), 3), "Q75": round(s.quantile(0.75), 3),
                    "Manquants": df[c].isnull().sum()
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    #  TAB 2: Correlations 
    with tab2:
        st.subheader("Analyse des Corrélations")
        if len(num_cols) < 2:
            st.info("Au moins 2 colonnes numériques requises.")
        else:
            method = st.radio("Méthode:", ["pearson", "spearman", "kendall"], horizontal=True)
            corr = df[num_cols].corr(method=method)

            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            title=f"Heatmap de corrélation ({method})", aspect="auto",
                            color_continuous_midpoint=0)
            fig.update_layout(height=550, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Corrélations les plus fortes")
            corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
            corr_pairs = [(i, j, corr.loc[i, j]) for (i, j), v in corr_unstacked.items()
                          if i != j and i < j]
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            corr_df = pd.DataFrame(corr_pairs[:15], columns=["Variable 1", "Variable 2", "Corrélation"])
            corr_df["Corrélation"] = corr_df["Corrélation"].round(4)
            st.dataframe(corr_df, use_container_width=True)

    #  TAB 3: Distributions 
    with tab3:
        st.subheader("Analyse des Distributions")
        chart_type = st.selectbox("Type de graphique:", [
            "Histogramme", "Boxplot", "Courbe de densité", "Scatter Plot", "Pie Chart", "Bar Chart"
        ])

        if chart_type == "Histogramme" and num_cols:
            col = st.selectbox("Colonne:", num_cols, key="hist_col")
            color_col = st.selectbox("Couleur par:", ["Aucune"] + cat_cols, key="hist_color")
            bins = st.slider("Nombre de bins:", 10, 100, 30)
            fig = px.histogram(df, x=col, nbins=bins, color=None if color_col == "Aucune" else color_col,
                               title=f"Distribution de {col}", template="plotly_dark",
                               color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boxplot":
            col = st.selectbox("Variable numérique:", num_cols, key="box_col")
            group = st.selectbox("Grouper par:", ["Aucune"] + cat_cols, key="box_group")
            fig = px.box(df, y=col, x=None if group == "Aucune" else group,
                         title=f"Boxplot — {col}", template="plotly_dark",
                         color=None if group == "Aucune" else group,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Courbe de densité" and num_cols:
            selected = st.multiselect("Colonnes:", num_cols, default=num_cols[:3], key="kde_cols")
            if selected:
                fig = go.Figure()
                colors = px.colors.qualitative.Bold
                for i, c in enumerate(selected):
                    data = df[c].dropna()
                    fig.add_trace(go.Violin(y=data, name=c, fillcolor=colors[i % len(colors)],
                                           line_color=colors[i % len(colors)], opacity=0.7))
                fig.update_layout(title="Courbes de densité (Violin)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot" and len(num_cols) >= 2:
            cx = st.selectbox("Axe X:", num_cols, key="sc_x")
            cy = st.selectbox("Axe Y:", num_cols, index=1 if len(num_cols) > 1 else 0, key="sc_y")
            color_col = st.selectbox("Couleur:", ["Aucune"] + cat_cols + num_cols, key="sc_color")
            fig = px.scatter(df, x=cx, y=cy,
                             color=None if color_col == "Aucune" else color_col,
                             title=f"Scatter: {cx} vs {cy}", template="plotly_dark",
                             trendline="ols", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart" and cat_cols:
            col = st.selectbox("Colonne catégorielle:", cat_cols, key="pie_col")
            top_n = st.slider("Top N valeurs:", 3, 15, 8)
            vc = df[col].value_counts().head(top_n)
            fig = px.pie(values=vc.values, names=vc.index, title=f"Répartition — {col}",
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart" and cat_cols:
            col = st.selectbox("Colonne catégorielle:", cat_cols, key="bar_col")
            vc = df[col].value_counts()
            fig = px.bar(x=vc.index, y=vc.values, title=f"Distribution — {col}",
                         labels={"x": col, "y": "Fréquence"}, template="plotly_dark",
                         color=vc.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    #  TAB 4: Missing Values 
    with tab4:
        st.subheader("Analyse des Valeurs Manquantes")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Colonne": missing.index,
            "Nombre manquants": missing.values,
            "Pourcentage (%)": missing_pct.values
        }).query("`Nombre manquants` > 0").sort_values("Nombre manquants", ascending=False)

        if missing_df.empty:
            st.success("Aucune valeur manquante dans le dataset!")
        else:
            st.dataframe(missing_df, use_container_width=True)

            fig = px.bar(missing_df, x="Colonne", y="Pourcentage (%)",
                         title="Pourcentage de valeurs manquantes par colonne",
                         template="plotly_dark", color="Pourcentage (%)",
                         color_continuous_scale="Reds")
            fig.add_hline(y=5, line_dash="dash", line_color="yellow",
                          annotation_text="Seuil 5%")
            st.plotly_chart(fig, use_container_width=True)

    #  TAB 5: Duplicates 
    with tab5:
        st.subheader("Analyse des Doublons")
        dupes = df[df.duplicated(keep=False)]
        n_dupes = df.duplicated().sum()
        c1, c2 = st.columns(2)
        c1.metric("Nombre de doublons", n_dupes)
        c2.metric("% du dataset", f"{n_dupes/len(df)*100:.2f}%")
        if n_dupes > 0:
            st.warning(f" {n_dupes} ligne(s) dupliquée(s) détectée(s).")
            st.dataframe(dupes.head(50), use_container_width=True)
        else:
            st.success("Aucun doublon détecté!")
