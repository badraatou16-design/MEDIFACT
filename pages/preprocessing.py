import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

def show():
    st.title("Preprocessing des Données")

    if st.session_state.df is None:
        st.warning("Aucun dataset chargé.")
        return

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Status bar
    missing_total = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes actuelles", f"{df.shape[0]:,}")
    c2.metric("Valeurs manquantes", missing_total)
    c3.metric("Doublons", dupes)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Valeurs manquantes", "Doublons", "Outliers",
        "Transformation", "Colonnes", "Filtrage"
    ])

    #  TAB 1: Missing Values 
    with tab1:
        st.subheader("Gestion des Valeurs Manquantes")
        missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        if not missing_cols:
            st.success("Aucune valeur manquante!")
        else:
            target_cols = st.multiselect("Colonnes à traiter:", missing_cols, default=missing_cols[:3])
            method = st.radio("Méthode:", [
                "Supprimer les lignes", "Supprimer les colonnes",
                "Remplacer par moyenne", "Remplacer par médiane",
                "Remplacer par mode", "Valeur personnalisée"
            ])
            custom_val = None
            if method == "Valeur personnalisée":
                custom_val = st.text_input("Valeur de remplacement:")

            if st.button("Appliquer", key="apply_missing") and target_cols:
                df_new = st.session_state.df.copy()
                for col in target_cols:
                    if method == "Supprimer les lignes":
                        df_new = df_new.dropna(subset=[col])
                    elif method == "Supprimer les colonnes":
                        df_new = df_new.drop(columns=[col])
                    elif method == "Remplacer par moyenne" and col in num_cols:
                        df_new[col].fillna(df_new[col].mean(), inplace=True)
                    elif method == "Remplacer par médiane" and col in num_cols:
                        df_new[col].fillna(df_new[col].median(), inplace=True)
                    elif method == "Remplacer par mode":
                        df_new[col].fillna(df_new[col].mode().iloc[0], inplace=True)
                    elif method == "Valeur personnalisée" and custom_val is not None:
                        df_new[col].fillna(custom_val, inplace=True)

                st.session_state.df = df_new
                st.session_state.history.append(f"Valeurs manquantes: {method} sur {target_cols}")
                st.success(f"Traitement appliqué! Dataset: {df_new.shape}")
                st.rerun()

    #  TAB 2: Duplicates 
    with tab2:
        st.subheader("Gestion des Doublons")
        st.metric("Doublons détectés", df.duplicated().sum())
        if st.button("Supprimer tous les doublons"):
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
            after = len(st.session_state.df)
            st.session_state.history.append(f"Doublons supprimés: {before - after} lignes")
            st.success(f" {before - after} doublons supprimés!")
            st.rerun()

    #  TAB 3: Outliers 
    with tab3:
        st.subheader("Gestion des Outliers")
        if not num_cols:
            st.info("Aucune colonne numérique.")
        else:
            col = st.selectbox("Colonne:", num_cols, key="out_col")
            method_out = st.radio("Méthode de détection:", ["IQR", "Z-Score"])

            data = st.session_state.df[col].dropna()
            if method_out == "IQR":
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:
                mean, std = data.mean(), data.std()
                lower, upper = mean - 3 * std, mean + 3 * std

            outliers = st.session_state.df[(st.session_state.df[col] < lower) | (st.session_state.df[col] > upper)]
            st.metric(f"Outliers détectés dans {col}", len(outliers))

            # Visualization before
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Avant suppression:**")
                fig = px.box(st.session_state.df, y=col, template="plotly_dark",
                             title=f"Boxplot — {col}")
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Supprimer les outliers"):
                df_clean = st.session_state.df[
                    (st.session_state.df[col] >= lower) & (st.session_state.df[col] <= upper)
                ].reset_index(drop=True)
                with c2:
                    st.markdown("**Après suppression:**")
                    fig2 = px.box(df_clean, y=col, template="plotly_dark",
                                  title=f"Boxplot — {col} (nettoyé)")
                    st.plotly_chart(fig2, use_container_width=True)
                st.session_state.df = df_clean
                st.session_state.history.append(f"Outliers supprimés: {col} ({method_out}) — {len(outliers)} lignes")
                st.success(f" {len(outliers)} outliers supprimés!")
                st.rerun()

    #  TAB 4: Transformations 
    with tab4:
        st.subheader("Transformation des Données")
        t_type = st.selectbox("Type de transformation:", [
            "Normalisation (Min-Max)", "Standardisation (Z-Score)",
            "Encodage Catégoriel", "Renommer une colonne", "Changer le type"
        ])

        if t_type in ["Normalisation (Min-Max)", "Standardisation (Z-Score)"]:
            cols_to_scale = st.multiselect("Colonnes à transformer:", num_cols)
            if st.button("Appliquer transformation") and cols_to_scale:
                df_new = st.session_state.df.copy()
                if t_type == "Normalisation (Min-Max)":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                df_new[cols_to_scale] = scaler.fit_transform(df_new[cols_to_scale])
                st.session_state.df = df_new
                st.session_state.history.append(f"{t_type}: {cols_to_scale}")
                st.success(f" {t_type} appliquée!")
                st.rerun()

        elif t_type == "Encodage Catégoriel":
            if not cat_cols:
                st.info("Aucune colonne catégorielle.")
            else:
                col_enc = st.selectbox("Colonne à encoder:", cat_cols)
                enc_method = st.radio("Méthode:", ["Label Encoding", "One-Hot Encoding"])
                if st.button("Encoder"):
                    df_new = st.session_state.df.copy()
                    if enc_method == "Label Encoding":
                        le = LabelEncoder()
                        df_new[col_enc + "_encoded"] = le.fit_transform(df_new[col_enc].astype(str))
                    else:
                        dummies = pd.get_dummies(df_new[col_enc], prefix=col_enc)
                        df_new = pd.concat([df_new, dummies], axis=1)
                    st.session_state.df = df_new
                    st.session_state.history.append(f"Encodage {enc_method}: {col_enc}")
                    st.success(f"Encodage appliqué!")
                    st.rerun()

        elif t_type == "Renommer une colonne":
            col_rename = st.selectbox("Colonne à renommer:", df.columns.tolist())
            new_name = st.text_input("Nouveau nom:")
            if st.button("Renommer") and new_name:
                st.session_state.df = st.session_state.df.rename(columns={col_rename: new_name})
                st.session_state.history.append(f"Colonne renommée: {col_rename} → {new_name}")
                st.success(f" '{col_rename}' renommée en '{new_name}'")
                st.rerun()

        elif t_type == "Changer le type":
            col_type = st.selectbox("Colonne:", df.columns.tolist())
            new_type = st.selectbox("Nouveau type:", ["int64", "float64", "str", "bool"])
            if st.button("Changer le type"):
                try:
                    st.session_state.df[col_type] = st.session_state.df[col_type].astype(new_type)
                    st.session_state.history.append(f"Type changé: {col_type} → {new_type}")
                    st.success(f"Type de '{col_type}' changé en {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")

    #  TAB 5: Column Selection 
    with tab5:
        st.subheader("Sélection / Suppression de Colonnes")
        cols_to_drop = st.multiselect("Colonnes à supprimer:", df.columns.tolist())
        if st.button("Supprimer les colonnes sélectionnées") and cols_to_drop:
            st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
            st.session_state.history.append(f"Colonnes supprimées: {cols_to_drop}")
            st.success(f" {len(cols_to_drop)} colonne(s) supprimée(s)")
            st.rerun()

        st.markdown("---")
        st.subheader("Sélectionner les colonnes utiles")
        keep_cols = st.multiselect("Colonnes à conserver:", df.columns.tolist(), default=df.columns.tolist())
        if st.button("Garder la sélection") and keep_cols:
            st.session_state.df = st.session_state.df[keep_cols]
            st.session_state.history.append(f"Colonnes conservées: {keep_cols}")
            st.success(f"Dataset réduit à {len(keep_cols)} colonnes")
            st.rerun()

    #  TAB 6: Filtering 
    with tab6:
        st.subheader("Filtrage Dynamique des Données")
        df_filtered = st.session_state.df.copy()

        for col in num_cols[:3]:
            if col in df_filtered.columns:
                min_v, max_v = float(df_filtered[col].min()), float(df_filtered[col].max())
                if min_v < max_v:
                    r = st.slider(f"Filtrer {col}:", min_v, max_v, (min_v, max_v), key=f"filt_{col}")
                    df_filtered = df_filtered[(df_filtered[col] >= r[0]) & (df_filtered[col] <= r[1])]

        for col in cat_cols[:2]:
            if col in df_filtered.columns:
                vals = st.multiselect(f"Filtrer {col}:", df_filtered[col].unique().tolist(),
                                      default=df_filtered[col].unique().tolist(), key=f"cat_{col}")
                df_filtered = df_filtered[df_filtered[col].isin(vals)]

        # Search
        search_col = st.selectbox("Rechercher dans:", df_filtered.columns.tolist(), key="search_col")
        search_val = st.text_input("Valeur à rechercher:")
        if search_val:
            df_filtered = df_filtered[df_filtered[search_col].astype(str).str.contains(search_val, case=False, na=False)]

        # Sort
        sort_col = st.selectbox("Trier par:", df_filtered.columns.tolist(), key="sort_col")
        sort_asc = st.radio("Ordre:", ["Ascendant", "Descendant"], horizontal=True) == "Ascendant"
        df_filtered = df_filtered.sort_values(sort_col, ascending=sort_asc)

        st.metric("Lignes après filtrage", len(df_filtered))
        st.dataframe(df_filtered.head(100), use_container_width=True)

        if st.button("Appliquer le filtrage au dataset"):
            st.session_state.df = df_filtered.reset_index(drop=True)
            st.session_state.history.append(f"Filtrage appliqué: {len(df_filtered)} lignes conservées")
            st.success("Dataset filtré appliqué!")
            st.rerun()

    # History
    st.markdown("---")
    if st.session_state.history:
        with st.expander("Historique des traitements"):
            for i, action in enumerate(st.session_state.history, 1):
                st.markdown(f"{i}. {action}")

    # Reset
    if st.button("↩ Réinitialiser au dataset original"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.history = ["Dataset réinitialisé"]
        st.success("Dataset réinitialisé!")
        st.rerun()
