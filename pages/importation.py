import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show():
    st.title("Importation des Données")
    st.markdown("Chargez votre dataset médical au format CSV.")

    # Demo datasets
    with st.expander("Charger un dataset de démonstration"):
        demo_choice = st.selectbox("Choisir un dataset démo:", [
            "Sélectionner...",
            "Diabète (Pima Indians)",
            "Maladies Cardiaques",
            "Données ICU Patients",
            "Données multi-sources synthétiques"
        ])
        if st.button("Charger le dataset démo") and demo_choice != "Sélectionner...":
            df = _generate_demo(demo_choice)
            st.session_state.df = df
            st.session_state.df_original = df.copy()
            st.session_state.history = [f"Dataset démo chargé: {demo_choice}"]
            st.success(f"Dataset '{demo_choice}' chargé avec succès!")

    st.markdown("---")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded:
        try:
            sep = st.selectbox("Séparateur:", [",", ";", "\t", "|"], index=0)
            encoding = st.selectbox("Encodage:", ["utf-8", "latin-1", "cp1252"], index=0)
            df = pd.read_csv(uploaded, sep=sep, encoding=encoding)
            st.session_state.df = df
            st.session_state.df_original = df.copy()
            st.session_state.history = [f"Fichier importé: {uploaded.name}"]
            st.success(f"Fichier '{uploaded.name}' importé avec succès!")
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        _display_dataset_info(df)

def _display_dataset_info(df):
    st.markdown("---")
    st.subheader("Aperçu du Dataset")

    # KPI metrics
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    dupes = df.duplicated().sum()
    missing = df.isnull().sum().sum()
    mem = df.memory_usage(deep=True).sum() / 1024

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lignes", f"{df.shape[0]:,}")
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Num. Colonnes", len(num_cols))
    c4.metric("Cat. Colonnes", len(cat_cols))
    c5.metric("Mémoire", f"{mem:.1f} KB")

    c1b, c2b = st.columns(2)
    c1b.metric("Valeurs manquantes", f"{missing:,}")
    c2b.metric("Doublons", dupes)

    # Data preview
    tabs = st.tabs(["Aperçu", "Statistiques", "Types", "Info"])

    with tabs[0]:
        n = st.slider("Nombre de lignes à afficher:", 5, 50, 10)
        st.dataframe(df.head(n), use_container_width=True)

    with tabs[1]:
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)

    with tabs[2]:
        type_df = pd.DataFrame({
            "Colonne": df.columns,
            "Type": df.dtypes.values,
            "Catégorie": ["Numérique" if c in df.select_dtypes(include=np.number).columns else "Catégorielle" for c in df.columns],
            "Valeurs uniques": [df[c].nunique() for c in df.columns],
            "Valeurs manquantes": [df[c].isnull().sum() for c in df.columns],
            "% Manquantes": [(df[c].isnull().sum()/len(df)*100).round(2) for c in df.columns]
        })
        st.dataframe(type_df, use_container_width=True)

    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Colonnes numériques:**")
            for c in df.select_dtypes(include=np.number).columns:
                st.markdown(f"- `{c}` ({df[c].dtype})")
        with col2:
            st.markdown("**Colonnes catégorielles:**")
            for c in df.select_dtypes(exclude=np.number).columns:
                st.markdown(f"- `{c}` ({df[c].dtype})")

def _generate_demo(name):
    np.random.seed(42)
    n = 300
    if "Diabète" in name:
        return pd.DataFrame({
            "Pregnancies": np.random.randint(0, 15, n),
            "Glucose": np.random.normal(120, 30, n).clip(0, 200).astype(int),
            "BloodPressure": np.random.normal(70, 12, n).clip(40, 120).astype(int),
            "SkinThickness": np.random.normal(25, 10, n).clip(0, 60).astype(int),
            "Insulin": np.random.exponential(80, n).clip(0, 500).astype(int),
            "BMI": np.random.normal(32, 7, n).clip(18, 60).round(1),
            "DiabetesPedigreeFunction": np.random.exponential(0.5, n).clip(0.05, 2.5).round(3),
            "Age": np.random.randint(21, 81, n),
            "Outcome": np.random.binomial(1, 0.35, n)
        })
    elif "Cardiaque" in name:
        return pd.DataFrame({
            "Age": np.random.randint(29, 77, n),
            "Sex": np.random.choice(["M", "F"], n),
            "ChestPain": np.random.choice(["typical", "atypical", "non-anginal", "asymptomatic"], n),
            "RestingBP": np.random.normal(130, 17, n).clip(90, 200).astype(int),
            "Cholesterol": np.random.normal(246, 52, n).clip(100, 600).astype(int),
            "FastingBS": np.random.binomial(1, 0.15, n),
            "MaxHR": np.random.normal(150, 23, n).clip(70, 210).astype(int),
            "ExerciseAngina": np.random.choice(["Y", "N"], n),
            "Oldpeak": np.random.exponential(1.0, n).clip(0, 6.2).round(1),
            "HeartDisease": np.random.binomial(1, 0.45, n)
        })
    elif "ICU" in name:
        df = pd.DataFrame({
            "PatientID": range(1001, 1001+n),
            "Age": np.random.randint(18, 90, n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "AdmissionType": np.random.choice(["Emergency", "Elective", "Urgent"], n),
            "ICU_LOS": np.random.exponential(3, n).clip(1, 30).round(1),
            "HeartRate": np.random.normal(85, 18, n).clip(40, 160).astype(int),
            "SystolicBP": np.random.normal(118, 22, n).clip(60, 200).astype(int),
            "Temperature": np.random.normal(37.0, 0.8, n).clip(35, 40.5).round(1),
            "SpO2": np.random.normal(96, 4, n).clip(75, 100).round(1),
            "GCS_Score": np.random.randint(3, 16, n),
            "Diagnosis": np.random.choice(["Sepsis", "Respiratory Failure", "Cardiac Arrest", "Trauma", "Stroke"], n),
            "Mortality": np.random.binomial(1, 0.2, n)
        })
        # Add some missing values
        for col in ["SpO2", "Temperature", "Cholesterol_Level"]:
            pass
        idx = np.random.choice(n, int(n*0.05), replace=False)
        df.loc[idx, "HeartRate"] = np.nan
        idx2 = np.random.choice(n, int(n*0.08), replace=False)
        df.loc[idx2, "Temperature"] = np.nan
        return df
    else:  # Multi-sources
        df = pd.DataFrame({
            "PatientID": range(1, n+1),
            "Age": np.random.randint(18, 85, n),
            "Gender": np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.49, 0.03]),
            "BMI": np.random.normal(26, 5, n).clip(15, 50).round(1),
            "Glucose": np.random.normal(100, 25, n).clip(60, 300).astype(int),
            "BloodPressure_Sys": np.random.normal(120, 20, n).clip(80, 200).astype(int),
            "BloodPressure_Dia": np.random.normal(80, 12, n).clip(50, 130).astype(int),
            "Cholesterol": np.random.normal(200, 40, n).clip(100, 400).astype(int),
            "HeartRate": np.random.normal(75, 15, n).clip(40, 150).astype(int),
            "HbA1c": np.random.normal(5.5, 1.2, n).clip(3.5, 14).round(1),
            "Creatinine": np.random.exponential(1.0, n).clip(0.3, 10).round(2),
            "WBC_Count": np.random.normal(7000, 2000, n).clip(2000, 20000).astype(int),
            "Hemoglobin": np.random.normal(13.5, 1.8, n).clip(7, 18).round(1),
            "SmokingStatus": np.random.choice(["Never", "Former", "Current"], n, p=[0.55, 0.25, 0.20]),
            "Comorbidity": np.random.choice(["Diabetes", "Hypertension", "CVD", "CKD", "None"], n),
            "Source": np.random.choice(["Hospital_A", "Hospital_B", "Clinic_C", "Lab_D"], n),
            "Outcome": np.random.choice(["Stable", "Improved", "Deteriorated"], n, p=[0.5, 0.3, 0.2])
        })
        # Inject missing values
        for col in ["BMI", "Glucose", "Cholesterol", "HbA1c", "Creatinine"]:
            idx = np.random.choice(n, int(n * np.random.uniform(0.03, 0.10)), replace=False)
            df.loc[idx, col] = np.nan
        # Inject duplicates
        dup_rows = df.sample(5, random_state=42)
        df = pd.concat([df, dup_rows], ignore_index=True)
        return df
