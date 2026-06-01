import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io

def show():
    st.title("Visualisations Interactives")

    if st.session_state.df is None:
        st.warning("Aucun dataset chargé.")
        return

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("** Options graphiques**")
        color_theme = st.selectbox("Palette:", [
            "Bold", "Vivid", "Safe", "Plotly", "D3", "G10"
        ], key="viz_palette")
        bgcolor = st.selectbox("Fond:", ["plotly_dark", "plotly", "ggplot2", "seaborn"])

    color_scale = getattr(px.colors.qualitative, color_theme, px.colors.qualitative.Bold)

    chart_type = st.selectbox("Type de graphique:", [
        "Histogramme interactif", "Boxplot", "Heatmap de corrélation",
        "Scatter Plot", "Scatter Matrix", "Line Chart",
        "Bar Chart", "Violin Plot", "Graphique comparatif", "Dashboard analytique"
    ])

    if chart_type == "Histogramme interactif" and num_cols:
        col = st.selectbox("Variable:", num_cols)
        color_col = st.selectbox("Couleur:", ["Aucune"] + cat_cols)
        bins = st.slider("Bins:", 10, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins,
                           color=None if color_col == "Aucune" else color_col,
                           marginal="box", template=bgcolor,
                           title=f"Distribution de {col}",
                           color_discrete_sequence=color_scale)
        st.plotly_chart(fig, use_container_width=True)
        _export_plotly(fig, f"hist_{col}")

    elif chart_type == "Boxplot" and num_cols:
        cols_sel = st.multiselect("Variables:", num_cols, default=num_cols[:4])
        group = st.selectbox("Grouper par:", ["Aucune"] + cat_cols)
        if cols_sel:
            if group == "Aucune":
                fig = go.Figure()
                for i, c in enumerate(cols_sel):
                    fig.add_trace(go.Box(y=df[c], name=c,
                                        marker_color=color_scale[i % len(color_scale)]))
                fig.update_layout(title="Boxplots comparatifs", template=bgcolor)
            else:
                col = cols_sel[0]
                fig = px.box(df, x=group, y=col, color=group,
                             title=f"{col} par {group}", template=bgcolor,
                             color_discrete_sequence=color_scale)
            st.plotly_chart(fig, use_container_width=True)
            _export_plotly(fig, "boxplot")

    elif chart_type == "Heatmap de corrélation" and len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Heatmap de Corrélation", template=bgcolor,
                        color_continuous_midpoint=0, aspect="auto")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        _export_plotly(fig, "heatmap_corr")

    elif chart_type == "Scatter Plot" and len(num_cols) >= 2:
        cx = st.selectbox("Axe X:", num_cols, key="sp_x")
        cy = st.selectbox("Axe Y:", num_cols, index=min(1, len(num_cols)-1), key="sp_y")
        color_col = st.selectbox("Couleur:", ["Aucune"] + cat_cols + num_cols, key="sp_c")
        size_col = st.selectbox("Taille:", ["Aucune"] + num_cols, key="sp_s")
        fig = px.scatter(df, x=cx, y=cy,
                         color=None if color_col == "Aucune" else color_col,
                         size=None if size_col == "Aucune" else df[size_col].clip(lower=0),
                         title=f"{cx} vs {cy}", template=bgcolor,
                         trendline="ols", opacity=0.8,
                         color_discrete_sequence=color_scale)
        st.plotly_chart(fig, use_container_width=True)
        _export_plotly(fig, f"scatter_{cx}_{cy}")

    elif chart_type == "Scatter Matrix" and len(num_cols) >= 2:
        sel = st.multiselect("Variables:", num_cols, default=num_cols[:4])
        color_col = st.selectbox("Couleur:", ["Aucune"] + cat_cols, key="sm_color")
        if sel:
            fig = px.scatter_matrix(df, dimensions=sel,
                                    color=None if color_col == "Aucune" else color_col,
                                    title="Scatter Matrix", template=bgcolor,
                                    color_discrete_sequence=color_scale)
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart" and num_cols:
        y_cols = st.multiselect("Variables Y:", num_cols, default=num_cols[:2])
        x_col = st.selectbox("Axe X:", df.columns.tolist(), key="lc_x")
        if y_cols:
            fig = px.line(df.head(200), x=x_col, y=y_cols,
                          title="Évolution temporelle", template=bgcolor,
                          color_discrete_sequence=color_scale)
            st.plotly_chart(fig, use_container_width=True)
            _export_plotly(fig, "linechart")

    elif chart_type == "Bar Chart" and cat_cols:
        cat = st.selectbox("Catégorie:", cat_cols, key="bc_cat")
        num = st.selectbox("Valeur:", ["Count"] + num_cols, key="bc_num")
        if num == "Count":
            vc = df[cat].value_counts().reset_index()
            vc.columns = [cat, "Count"]
            fig = px.bar(vc, x=cat, y="Count", title=f"Fréquences — {cat}",
                         template=bgcolor, color="Count",
                         color_continuous_scale="Blues")
        else:
            agg = df.groupby(cat)[num].mean().reset_index()
            fig = px.bar(agg, x=cat, y=num, title=f"Moyenne {num} par {cat}",
                         template=bgcolor, color=num,
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        _export_plotly(fig, "barchart")

    elif chart_type == "Violin Plot" and num_cols:
        col = st.selectbox("Variable:", num_cols, key="vp_col")
        group = st.selectbox("Grouper par:", ["Aucune"] + cat_cols, key="vp_grp")
        fig = px.violin(df, y=col, x=None if group == "Aucune" else group,
                        box=True, points="outliers", template=bgcolor,
                        title=f"Violin Plot — {col}",
                        color=None if group == "Aucune" else group,
                        color_discrete_sequence=color_scale)
        st.plotly_chart(fig, use_container_width=True)
        _export_plotly(fig, "violin")

    elif chart_type == "Graphique comparatif" and len(num_cols) >= 2:
        col1_cmp = st.selectbox("Variable 1:", num_cols, key="cmp1")
        col2_cmp = st.selectbox("Variable 2:", num_cols, index=1, key="cmp2")
        group_cmp = st.selectbox("Grouper par:", ["Aucune"] + cat_cols, key="cmp_grp")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col1_cmp, title=f"Distribution — {col1_cmp}",
                               template=bgcolor, color_discrete_sequence=[color_scale[0]])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x=col2_cmp, title=f"Distribution — {col2_cmp}",
                               template=bgcolor, color_discrete_sequence=[color_scale[1]])
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Dashboard analytique":
        st.subheader("Dashboard Analytique — Vue d'Ensemble")
        if num_cols:
            c1, c2, c3, c4 = st.columns(4)
            for i, col in enumerate(num_cols[:4]):
                [c1, c2, c3, c4][i].metric(col, f"{df[col].mean():.2f}",
                                             delta=f"σ={df[col].std():.2f}")

            fig_grid = []
            selected_num = num_cols[:4]
            for col in selected_num:
                fig = px.histogram(df, x=col, nbins=30, template=bgcolor,
                                   title=f"Distribution — {col}",
                                   color_discrete_sequence=[color_scale[selected_num.index(col) % len(color_scale)]])
                fig.update_layout(height=300, showlegend=False)
                fig_grid.append(fig)

            cols_grid = st.columns(2)
            for i, fig in enumerate(fig_grid):
                with cols_grid[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig_corr = px.imshow(corr, text_auto=".1f", color_continuous_scale="RdBu_r",
                                     title="Corrélations", template=bgcolor, aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)


def _export_plotly(fig, name):
    buf = io.StringIO()
    fig.write_html(buf)
    st.download_button(
        f"Télécharger le graphique (HTML)",
        buf.getvalue(),
        file_name=f"{name}.html",
        mime="text/html",
        key=f"dl_{name}_{id(fig)}"
    )
