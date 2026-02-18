import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## To run this code
    `uv pip install marimo` <br>
    `marimo edit (file.py)`

    Dependencies
    `uv pip install morethemes pyfonts pypalettes drawarrow`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #_EXPLORING CCR DATA_
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## IMPORTS
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import contextily as ctx
    import geopandas as gpd
    import marimo as mo
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import matplotlib.ticker as ticker
    import matplotlib.colors as mcolors

    import imageio
    import os
    import shutil

    from shapely.affinity import translate, scale

    import morethemes as mt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from drawarrow import fig_arrow

    # Fonts, Palettes, Themes, Arrow
    from pyfonts import load_google_font, load_font
    from pypalettes import load_palette, load_cmap
    import dayplot as dp

    import hdbscan
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
    from sklearn.decomposition import PCA, FastICA
    return (
        Path,
        dp,
        fig_arrow,
        gpd,
        imageio,
        load_cmap,
        load_font,
        load_google_font,
        matplotlib,
        mcolors,
        mdates,
        mo,
        mpatches,
        mt,
        mtick,
        np,
        os,
        pd,
        plt,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## FUNCTIONS & SETUP
    """)
    return


@app.cell
def _(Path, load_google_font, mt):
    font = load_google_font("Roboto", italic=True)
    mt.set_theme("nature")

    current_dir = Path.cwd()
    seeds_dir = current_dir / "data" / "dbt_pipeline" / "seeds"

    source_text = "Source: CCR (Caisse Centrale de Réassurance)"
    return font, seeds_dir, source_text


@app.cell
def _(formats, mdates, pd, plt, source_text):
    def global_evolution_catnat(df, dropdown, font=None, source_text=source_text):
        df["dateArrete"] = pd.to_datetime(df["dateArrete"], format="%d-%m-%Y")

        df_grouped = (
            df.set_index("dateArrete")
            .resample(dropdown.selected_key)
            .size()
            .reset_index(name="count")
            .set_index("dateArrete")
            .loc[:"2025"]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(
            df_grouped.index,
            df_grouped["count"],
            linestyle="-",
            color="black",
        )

        plt.title("Evolution des Arrêtés", 
                  font=font, 
                  fontsize=20, 
                  color="black", 
                  pad=20)

        plt.text(
            0.5,
            1.01,
            "(Global)",
            font=font,
            fontsize=11,
            color="black",
            ha="center",
            transform=ax.transAxes,
        )
        plt.xlabel("")
        plt.ylabel("Occurances", font=font, fontsize=14, color="black")

        ax.yaxis.set_label_coords(-0.04, 0.65)

        format = formats[dropdown.selected_key]

        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(format))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=30))

        plt.xticks(font=font, rotation=15, fontsize=9)
        plt.yticks(font=font, fontsize=10)
        plt.xlim([df_grouped.index.min(), pd.to_datetime("2025-12-31")])

        plt.text(
            1,
            -0.105,
            source_text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            font=font,
        )

        plt.tight_layout()
        return ax
    return (global_evolution_catnat,)


@app.cell
def _(plt, source_text):
    def bar_distribution(df, 
                         font=None,
                         figsize=(9, 5),
                         filter_year=1983,
                         color="#C4C4C4",
                         analysis="reconnaissance"):

        df_plot = df.copy()
        df_plot = df_plot[df_plot["dateArrete"].dt.year > filter_year]

        if analysis == 'franchise':
            selected_col = "franchise"
            df_plot = df_plot[
                df_plot["libelleAvis"] == "Reconnue"
            ]
            title = "Répartition des Franchises"
        else:
            selected_col = "libelleAvis"
            title = "Répartition des Avis"

        df_plot = (
            df_plot.groupby(selected_col).count().sort_values(by="nomCommune")
        )

        _, ax = plt.subplots(figsize=figsize)
        bars = plt.barh(
            df_plot.index, df_plot["nomCommune"], color=color, height=0.55
        )

        plt.title(title, font=font, fontsize=20, color="black", pad=20)
        plt.text(
            0.5,
            1.02,
            f"(Par Communes après {filter_year})",
            font=font,
            fontsize=10,
            color="black",
            ha="center",
            transform=ax.transAxes,
        )  # subtitle

        plt.ylabel("")

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)

        # Add labels to each bar
        for bar in bars:
            width = bar.get_width()
            plt.gca().text(
                width + 1500,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(width):,}".replace(",", "'"),
                ha="left",
                va="center",
                color="black",
                font=font,
            )

        plt.yticks(font=font, fontsize=11)

        if analysis == "reconnaissance":
            percent_nr = (
                df_plot["nomCommune"].loc["Non reconnue"]
                / df_plot["nomCommune"].sum()
            )

            plt.text(
                0.65,
                0.45,
                f"Le pourcentage d'arrêtés $\mathbf{{Non\ Reconnue}}$ est de $\mathbf{{{(percent_nr * 100).round(1)}}}$%",
                font=font,
                fontsize=10,
                color="black",
                ha="center",
                transform=ax.transAxes,
            )

        plt.text(
                1,
                -0.075,
                source_text,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                font=font,
                )
        plt.grid(False)
        plt.tight_layout()

        return ax
    return (bar_distribution,)


@app.cell
def _(plt, source_text):
    def bar_total_evolution(df,
                            title="Evolution Non-Reconnaissance",
                            filter_col="libelleAvis", 
                            split_year=2010,
                            font=None):

        split_period = f'periode_{split_year}'
        if filter_col == "libelleAvis":
            target_col = "Non reconnue"
        else:
            target_col = "Doublée ou plus"

        total_per_period = df.groupby(split_period)['count'].sum().reset_index()

        non_rec_per_period = df[df[filter_col] == target_col]\
                            .groupby(split_period)['count'].sum().reset_index()

        plot_df = total_per_period.rename(columns={'count': 'Total'}).merge(
            non_rec_per_period.rename(columns={'count': target_col}), on=split_period, how='left'
        ).fillna(0)

        plot_df = plot_df.sort_values(target_col)

        pct_labels = [f"{(non / tot * 100):.0f}%" if tot > 0 else "0%" 
                      for non, tot in zip(plot_df[target_col], plot_df['Total'])]

        fig, ax = plt.subplots(figsize=(8, 6))

        bar_total = ax.bar(plot_df[split_period], plot_df['Total'], 
                            color='lightgrey', 
                            label='Total', 
                            edgecolor='darkgrey',
                            linewidth=1,
                            width=0.42)

        bar_non = ax.bar(plot_df[split_period], 
                          plot_df[target_col], 
                          color='#AA4A44', 
                          label=target_col,
                          linewidth=1,
                          width=0.4)

        ax.bar_label(bar_total, padding=1, 
                     fontproperties=font, 
                     fontsize=12, 
                     color='black',
                     fontweight='bold')

        ax.bar_label(bar_non, labels=pct_labels, 
                     label_type='edge', 
                     fontproperties=font,
                     color='black', 
                     fontsize=14, 
                     fontweight='bold')

        plt.title(title, fontproperties=font, fontsize=22, pad=20, color="black")
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("")

        plt.legend(frameon=False, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.1), 
                   prop=font,
                   fontsize=11, 
                   labelcolor='black')

        plt.text(
                1,
                -0.125,
                source_text,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                font=font,
            )

        ax.grid(False)
        ax.set_xlim(-0.4, 1.4)
        ax.tick_params(axis='both', which='both', length=0)

        plt.xticks(font=font, fontsize=14)
        ax.tick_params(axis='x', pad=10)

        plt.tight_layout()

        return ax
    return (bar_total_evolution,)


@app.cell
def _(mdates, pd, plt, source_text):
    def prepare_data_line_chart(df, 
                                group_freq="YE",
                                target_col="libelleAvis",
                                end_date="2025-12-31", 
                                remove_simple=True):
        df_plot = df.copy()
        if target_col == "franchise":
            df_plot = df_plot[(df_plot["libelleAvis"] != "Non reconnue") & \
                              ((df_plot[target_col] != "Simple") if remove_simple else True)]

        df_plot = (
            df_plot.groupby(
                [pd.Grouper(key="dateArrete", freq=group_freq), target_col]
            )
            .size()
            .reset_index(name="count")
        )

        df_plot = df_plot[
            df_plot["dateArrete"].dt.year <= 2026
        ]

        df_plot = df_plot.pivot(index="dateArrete", columns=target_col, values="count").fillna(0)
        df_plot = df_plot.loc[:end_date]

        return df_plot

    def line_chart_global(df, 
                          font=None,
                          group_freq="YE",
                          target_col="libelleAvis",
                          start_date="1984",
                          end_date="2025-12-31"):

        annee = int(group_freq.split("YE")[0] or 1)
        df_plot = prepare_data_line_chart(df, 
                                          group_freq=group_freq,
                                          target_col=target_col,
                                          end_date=end_date)
        if target_col == "libelleAvis":
            title = "Evolution des Avis"
            subtitle = f"(moyenne sur {annee} ans)"
        else:
            title = "Evolution des Franchises"
            subtitle = f"(exclus franchise simple, moyenne sur {annee} ans)"

        _, ax = plt.subplots(figsize=(10, 6))

        for column in df_plot.columns:
            if target_col == "libelleAvis":
                    if "Non reconnue" in column:
                        color = "#e63946"
                        linewidth = 3
                        zorder = 5

                        ax.plot(
                            df_plot.index,
                            df_plot[column],
                            label=column,
                            color=color,
                            linewidth=linewidth,
                            marker="o",
                            linestyle="-",
                            ms=5,
                            markerfacecolor="white",
                            markeredgewidth=1,
                            zorder=zorder,
                        )
                    else:
                        ax.plot(
                            df_plot.index,
                            df_plot[column],
                            label=column,
                            marker="o",
                            linestyle="-",
                            ms=5,
                            markerfacecolor="white",
                            markeredgewidth=1,
                            alpha=0.75,
                            zorder=1,
                        )
            else:
                 ax.plot(
                        df_plot.index,
                        df_plot[column],
                        label=column,
                        marker="o",
                        linestyle="-",
                        ms=5,
                        markerfacecolor="white",
                        markeredgewidth=1,
                        alpha=1,
                        zorder=1,
                    )

        plt.title(title, font=font, fontsize=20, color="black", pad=20)
        plt.text(
            0.5,
            1.01,
            subtitle,
            font=font,
            fontsize=11,
            color="black",
            ha="center",
            transform=ax.transAxes,
        )

        plt.xlabel("")
        plt.ylabel("Occurrences", font=font, fontsize=14, color="black")
        ax.yaxis.set_label_coords(-0.06, 0.65)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=30))

        plt.xticks(font=font, rotation=15, fontsize=9)
        plt.yticks(font=font, fontsize=10)

        plt.xlim([pd.to_datetime(start_date), pd.to_datetime("2026-03-31")])

        ax.legend(prop=font, loc="upper right", 
                  frameon=False, labelcolor="black")

        handles, labels = ax.get_legend_handles_labels()

        if target_col == "franchise":
            desired_order = ["Doublée", "Triplée", "Quadruplée"]

            order_map = dict(zip(labels, handles))
            new_handles = [order_map[l] for l in desired_order if l in order_map]
            new_labels = [l for l in desired_order if l in order_map]

            ax.legend(new_handles, new_labels, prop=font, loc="upper left", 
                      frameon=False, labelcolor="black")

        plt.text(
            1,
            -0.125,
            source_text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            font=font,
        )

        plt.tight_layout()

        return ax
    return line_chart_global, prepare_data_line_chart


@app.cell
def _(fig_arrow, mdates, mtick, pd, plt, prepare_data_line_chart, source_text):
    def add_arrow_indicators(ax, df, font=None, indicators_config=None):
        if indicators_config is None:
            indicators_config = {
                "Maximum": {
                    "val": df.max(),
                    "head": (0.975, 0.615),
                    "tail": (0.88, 0.7),
                    "radius": 0.1,
                    "text_pos": (0.875, 0.74)
                },
                "Minimum": {
                    "val": df[df > 0].min(),
                    "head": (0.6475, 0.19),
                    "tail": (0.7, 0.15),
                    "radius": -0.1,
                    "text_pos": (0.81, 0.015)
                }
            }

        for label, cfg in indicators_config.items():
            fig_arrow(
                head_position=cfg["head"],
                tail_position=cfg["tail"],
                radius=cfg["radius"],
                width=1, 
                color="black",
                fill_head=False, 
                mutation_scale=0.75, 
            )

            ax.text(
                cfg["text_pos"][0], 
                cfg["text_pos"][1],
                f"{label}: {cfg['val']*100:.1f}%",
                ha="right", 
                va="bottom", 
                transform=ax.transAxes,
                fontsize=11, 
                color="black", 
                fontproperties=font
            )

    def line_chart_taux(df, font=None, 
                        group_freq="YE", 
                        target_col="libelleAvis", 
                        end_date="2025-12-31", 
                        indicators_positions=None):

        df_plot = df.copy()
        if target_col == "libelleAvis":
            select_cols = ["Non reconnue", "Reconnue"]
            title="Taux de Non Reconnaissance"
        else:
            df_plot['franchise'] = df_plot["franchise"].replace(
                                                              {"Doublée":"Doublée ou plus",
                                                               "Triplée":"Doublée ou plus",
                                                               "Quadruplée":"Doublée ou plus"})
            select_cols = ["Doublée ou plus", "Simple"]
            title="Taux de Franchises 'Doublée ou plus'"

        df_plot = prepare_data_line_chart(
            df_plot, group_freq=group_freq, target_col=target_col, end_date=end_date, remove_simple=False
        )

        total = df_plot[select_cols[0]] + df_plot[select_cols[1]]
        df_plot = df_plot[select_cols[0]] / total.replace(0, 1)
        df_plot = df_plot.loc[:end_date]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F5F5DC')

        ax.plot(
            df_plot.index, df_plot,
            color="#e63946", linewidth=3, marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1
        )

        if target_col == "libelleAvis":
            add_arrow_indicators(ax, df_plot, font=font, indicators_config=indicators_positions)

        ax.set_title(title, fontproperties=font, fontsize=18, color="black")
        ax.set_ylabel("Taux", fontproperties=font, fontsize=14, color="black")
        ax.yaxis.set_label_coords(-0.06, 0.65)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=30))
        plt.xticks(rotation=15, fontsize=9, fontproperties=font)
        ax.set_xlim([df_plot.index.min(), pd.to_datetime("2026-03-31")])

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.0, 1.0)
        plt.yticks(fontsize=10, fontproperties=font)

        ax.legend(prop=font, loc="upper right", frameon=False)
        ax.text(
            1, -0.125, source_text, ha="right", va="bottom", 
            transform=ax.transAxes, fontsize=8, color="black", fontproperties=font
        )

        # Clean Spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        return ax
    return (line_chart_taux,)


@app.cell
def _(plt, source_text):
    def donut_global_type_catnat(df, font=None):

        df_plot = df.copy()
        peril_order = ['Inondations', 'Sécheresse', 'Autre']

        def get_ordered_counts(df, order):
            counts = df['nomPeril'].value_counts().reindex(order)
            return counts.values, counts.index

        sizes_total, labels_total = get_ordered_counts(df, peril_order)

        df_avant = df_plot[df_plot['periode_2010'] == "Avant 2010"]
        sizes_avant, labels_avant = get_ordered_counts(df_avant, peril_order)

        df_apres = df_plot[df_plot['periode_2010'] == "Après 2010"]
        sizes_apres, labels_apres = get_ordered_counts(df_apres, peril_order)

        sizes = [sizes_total, sizes_avant, sizes_apres]
        labels = [labels_total, labels_avant, labels_apres]

        layout = """
            AAA
            B.C
        """
        fig, ax_dict = plt.subplot_mosaic(layout, figsize=(10, 6))

        def draw_donut(ax, sizes, labels, title, colors, main=False):
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', 
                   startangle=70, pctdistance=0.80, textprops={'fontproperties': font, "color":"black", "fontsize":12})

            centre_circle = plt.Circle((0,0), 0.6, fc='#F5F5DC')
            ax.add_artist(centre_circle)

            ax.axis('equal')
            ax.set_title(title, fontproperties=font, fontsize=14, color="black")


        colors_main = ['#496988', '#AA4A44', 'lightgrey', '#ffcc99']
        draw_donut(ax_dict['A'], sizes[0], labels[0], "Période: 1982 - 2025", colors_main)
        draw_donut(ax_dict['B'], sizes[1], labels[1], "Avant 2010", colors_main)
        draw_donut(ax_dict['C'], sizes[2], labels[2], "Après 2010", colors_main)

        fig.text(
            0.95,          
            0.0,          
            source_text,
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
            font=font,
        )

        plt.suptitle("Répartition par CatNat", font=font, color="black", fontsize=22)


        plt.tight_layout()

        return fig
    return (donut_global_type_catnat,)


@app.cell
def _(plt, source_text):
    def donut_specific(df, 
                       target_col="libelleAvis",
                       colors=['lightgrey', '#496988', '#AA4A44']):

        df_plot = df.copy()

        if target_col == "libelleAvis":
            title = "Répartition des Non Reconnaissances"
            df_plot = df_plot[df_plot["libelleAvis"] == "Non reconnue"]
        else:
            df_plot = df_plot[(df_plot[target_col] != "Simple") & (df_plot["libelleAvis"] == "Reconnue")]
            title = "Répartition des Franchises 'Doublée ou plus'"


        counts = df_plot.groupby("nomPeril").size()
        sizes = counts.values
        labels_ = counts.index

        fig, ax = plt.subplots(figsize=(8, 6))

        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels_, 
            colors=colors, 
            autopct='%1.0f%%', 
            startangle=50, 
            pctdistance=0.80, 
            textprops={"color": "black", "fontsize": 14}
        )

        centre_circle = plt.Circle((0,0), 0.6, fc='#F5F5DC')
        ax.add_artist(centre_circle)

        ax.axis('equal') 
        ax.set_title(title, 
                     fontsize=18, color="black", pad=20)

        plt.text(
                0.975,
                -0.125,
                source_text,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
            )

        plt.tight_layout()

        return ax
    return (donut_specific,)


@app.cell
def _(mpatches, np, pd, plt, source_text):
    def stacked_area_type_catnat(df, 
                                 font=None, 
                                 target_col="libelleAvis",
                                 hatch='//',
                                 limit_years=[1984, 2025],
                                 window=3):

        df_plot = df.copy()
        if target_col == "libelleAvis":
            title = "Evolution par CatNat et par Reconnaissance"
            status_request = 'Non reconnue'

            custom_order = ['Reconnue,Inondations', 'Non reconnue,Inondations',
                            'Reconnue,Sécheresse', 'Non reconnue,Sécheresse',
                            'Reconnue,Autre', 'Non reconnue,Autre']
        else:
            df_plot['franchise'] = np.where(
                                            df_plot['franchise'] == 'Simple', 
                                            'Simple', 
                                            'Doublée ou plus'
                                            )
            title = "Evolution par CatNat et par Franchise"
            status_request = 'Doublée ou plus'

            custom_order = ['Simple,Inondations', 'Doublée ou plus,Inondations',
                            'Simple,Sécheresse', 'Doublée ou plus,Sécheresse',
                            'Simple,Autre', 'Doublée ou plus,Autre']

        df_plot = df.pivot_table(
            index='dateArrete', 
            columns=[target_col, 'nomPeril'],
            values='count', 
            aggfunc='sum'
        ).fillna(0)

        df_plot = df_plot.rolling(window=window, center=False).mean()



        new_columns = [(c.split(",")[0], c.split(",")[1]) for c in custom_order]
        df_plot = df_plot[new_columns]

        _, ax = plt.subplots(figsize=(10, 6))

        x = df_plot.index
        y_values = [df_plot[col] for col in df_plot.columns]
        labels = [f"{p} ({s})" for s, p in df_plot.columns]

        layers = ax.stackplot(x, y_values, 
                                 labels=labels, 
                                 alpha=0.8, 
                                 colors=["#496988", "#496988", "#AA4A44", "#AA4A44", "lightgrey", "lightgrey"],
                                 edgecolor="#19242E",
                                 linewidth=0.3)

        for i, (status, peril) in enumerate(df_plot.columns):
            if status == status_request:
                layers[i].set_hatch(hatch)
                layers[i].set_alpha(0.5)

        peril_handles = [
            mpatches.Patch(color="#496988", label='Inondations'),
            mpatches.Patch(color="#AA4A44", label='Sécheresse'),
            mpatches.Patch(color="lightgrey", label='Autre')
        ]

        status_handles = [
            mpatches.Patch(facecolor='darkgrey', label='Reconnue' if target_col == "libelleAvis" else "Simple"),
            mpatches.Patch(facecolor='darkgrey', hatch=hatch, edgecolor='white', label='Non reconnue' if target_col == "libelleAvis" else "Doublée ou plus")
        ]

        legend_peril = ax.legend(handles=peril_handles, loc='upper left', 
                                    bbox_to_anchor=(0.05, 1), title="", 
                                    prop=font,
                                    fontsize=10, labelcolor='black',
                                    frameon=False)
        ax.add_artist(legend_peril)

        ax.legend(handles=status_handles, loc='upper left', 
                     bbox_to_anchor=(0.2, 1), title="",
                     prop=font,
                     fontsize=10, labelcolor='black',
                     frameon=False)
        ax.grid(False)

        plt.title(title, font=font, fontsize=20, color="black", pad=20)
        plt.text(
            0.5,
            1.0,
            f"(Moyenne mobile sur {window} ans)",
            font=font,
            fontsize=10,
            color="black",
            ha="center",
            transform=ax.transAxes,
        ) 

        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')

        ax.set_ylabel("Nombre d'arrêtés", font=font, fontsize=13, color="black")
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)

        ax.yaxis.set_label_coords(1.06, 0.65)


        plt.yticks(font=font, fontsize=11)
        plt.xticks(font=font, fontsize=11)

        plt.text(
            1,
            -0.125,
            source_text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            font=font,
        )

        x_start = pd.to_datetime(str(limit_years[0]+window))
        x_end = pd.to_datetime(str(limit_years[1]))
        plt.xlim([x_start, x_end])
        plt.tight_layout()

        return ax
    return (stacked_area_type_catnat,)


@app.cell
def _(plt, source_text):
    def bar_catnat_evolution(df,
                             target_col="libelleAvis",
                             perils=["Inondations", "Sécheresse"],
                             split_year=2010):

        split_period = f'periode_{split_year}'

        if target_col == "libelleAvis":
            select_cols = ["Non reconnue", "Reconnue"]
            title="Evolution Non-Reconnaissance"
        else:
            select_cols = ["Doublée ou plus", "Simple"]
            title="Evolution des Franchises"

        df_filtered = df[df["nomPeril"].isin(perils)]
        df_plot = df_filtered.groupby(['nomPeril', split_period, target_col])['count'].sum().unstack(fill_value=0)
        df_plot['Total'] = df_plot[select_cols[0]] + df_plot[select_cols[1]]
        df_plot = df_plot.reset_index()
        df_plot = df_plot.sort_values(select_cols[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

        for i, peril in enumerate(perils):
            ax = axes[i]
            data = df_plot[df_plot['nomPeril'] == peril]

            bar_tot = ax.bar(data[split_period], data['Total'], 
                             color='lightgrey', 
                              edgecolor='darkgrey', width=0.65)

            bar_non = ax.bar(data[split_period], data[select_cols[0]], 
                             color='#AA4A44', linewidth=1.5, width=0.6)

            ax.bar_label(bar_tot, padding=5, fontsize=10, fontweight='bold', color="black")

            pcts = [f"{(n/t*100):.0f}%" if t > 0 else "0%" for n, t in zip(data[select_cols[0]], data['Total'])]
            ax.bar_label(bar_non, labels=pcts, label_type='edge',
                          color='black', fontsize=11, fontweight='bold')

            ax.set_title(peril, fontsize=16, pad=15, color="black")
            ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=12)

            ax.spines[['top', 'right', 'bottom'] if peril == "Sécheresse" else ['top', 'right', "left", 'bottom']].set_visible(False)
            ax.yaxis.set_visible(False)

            ax.grid(False)

        handles = [plt.Rectangle((0,0),1,1, color='lightgrey'), 
                   plt.Rectangle((0,0),1,1, color='#AA4A44')]
        fig.legend(handles, ['Total', select_cols[0]], 
                   loc='upper center', bbox_to_anchor=(0.65, 0.8), 
                   ncol=1, 
                   frameon=False,
                   fontsize=11, 
                   labelcolor='black')

        plt.text(
                0.975,
                -0.125,
                source_text,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
            )

        plt.suptitle(title, fontsize=20, color="black")

        plt.tight_layout()

        return fig
    return (bar_catnat_evolution,)


@app.cell
def _(mo):
    mo.md(r"""
    ## _READING MAIN DATA_
    """)
    return


@app.cell
def _(pd, seeds_dir):
    main_data = pd.read_csv(seeds_dir / "ccr_main_page.csv")
    main_data
    return (main_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ### _TIME EVOLUTION OF DISASTERS_
    """)
    return


@app.cell
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["ME", "YE"], value="YE", label="Select Frequency"
    )

    formats = {"ME": "%Y-%m", "YE": "%Y"}
    dropdown
    return dropdown, formats


@app.cell
def _(dropdown, global_evolution_catnat, main_data, mo):
    mo.mpl.interactive(global_evolution_catnat(main_data, dropdown, font=None))
    return


@app.cell
def _(main_data):
    main_data_occur_total = main_data.groupby("dateArrete").size()

    main_data["nomPeril"] = (
        main_data["nomPeril"]
        .mask(main_data["nomPeril"].str.contains("Inondations"), "Inondations")
        .where(main_data["nomPeril"].str.contains("Inondations|Sécheresse"), "Autre")
    )

    main_data_occur_inond = main_data[main_data["nomPeril"] == "Inondations"].groupby("dateArrete").size()
    main_data_occur_secheresse = main_data[main_data["nomPeril"] == "Sécheresse"].groupby("dateArrete").size()
    return (
        main_data_occur_inond,
        main_data_occur_secheresse,
        main_data_occur_total,
    )


@app.cell
def _(
    dp,
    font,
    load_font,
    main_data_occur_inond,
    main_data_occur_secheresse,
    main_data_occur_total,
    pd,
    plt,
    source_text,
):
    from matplotlib.colors import LinearSegmentedColormap

    accent_color = "#e63946" 
    cmap = LinearSegmentedColormap.from_list("monochrome", [accent_color, accent_color])

    font_url = "https://github.com/coreyhu/Urbanist/blob/main/fonts/ttf"
    fontyear = load_font(f"{font_url}/Urbanist-Medium.ttf?raw=true")

    start_year = 2000
    end_year = 2025
    years_to_plot = range(start_year, end_year + 1)

    colors_band = ["black", "blue", "red"]
    cmaps = [LinearSegmentedColormap.from_list("monochrome", [colors_band[0], colors_band[0]]),
             LinearSegmentedColormap.from_list("monochrome", [colors_band[1], colors_band[1]]),
             LinearSegmentedColormap.from_list("monochrome", [colors_band[2], colors_band[2]])]

    style_args = dict(
        day_kws={"alpha": 0},          # Hide day borders for smoother blending
        month_kws={"font": fontyear, "size": 10},
        month_y_margin=1,
        color_for_none="None",         # Important: None/Transparent background for stacking
        alpha=0.3
    )

    data = [main_data_occur_total, main_data_occur_inond, main_data_occur_secheresse]
    titles = ["Total", "Inondations", "Sécheresse"]
    fig, axs = plt.subplots(nrows=3, figsize=(15, 5))

    for i, (df_cal, color, title) in enumerate(zip(data, colors_band, titles)):
        for _, year in enumerate(years_to_plot):
            subset = df_cal[df_cal.index.year == year].copy()

            if subset.empty: continue

            try:
                delta_years = end_year - year
                subset.index = subset.index + pd.offsets.DateOffset(years=delta_years)
            except ValueError:
                subset = subset[~((subset.index.month == 2) & (subset.index.day == 29))]
                subset.index = subset.index.map(lambda d: d.replace(year=end_year))

            layer_style = style_args.copy()
            layer_style['cmap'] = cmaps[i]
            layer_style['month_kws']["color"] = color

            dp.calendar(
                subset.index,
                subset.values,
                start_date=f"{end_year}-01-01",
                end_date=f"{end_year}-12-31",
                ax=axs[i],
                **layer_style
            )

        axs[i].text(
            x=-2, y=3.5, 
            s=title, 
            size=13, rotation=90, 
            color=color, 
            va="center", font=fontyear
        )

        axs[i].set_facecolor("#FBFBF6")

    plt.suptitle("Récurrencences des CatNat (2000-2025)", fontsize=20, color="black")

    plt.text(
            1,
            -0.125,
            source_text,
            ha="right",
            va="bottom",
            transform=axs[-1].transAxes,
            fontsize=8,
            color="black",
            font=font,
        )

    plt.show()
    return (LinearSegmentedColormap,)


@app.cell
def _(mo):
    mo.md(r"""
    ## _READING DETAILS DATA_
    """)
    return


@app.cell
def _(pd, seeds_dir):
    details_data = pd.read_csv(seeds_dir / "ccr_details.csv")
    details_data["dateArrete"] = pd.to_datetime(details_data["dateArrete"])
    details_data["libelleAvis"] = details_data["libelleAvis"].replace(
        {
            "Reconnue(sans impact sur la modulation)": "Reconnue"
        }
    )

    details_data = details_data[~details_data.duplicated()] # Error dans CCR https://www.ccr.fr/detail-arrete/?codeArrete=000888

    details_data.head(5) # Use this comullative
    return (details_data,)


@app.cell
def _(details_data):
    details_data[details_data["franchise"] != '-'].groupby(["codeInsee","nomCommune","dateArrete", "franchise"]).size()
    return


@app.cell
def _(details_data):
    details_data[(details_data["franchise"] != '-') &\
                 (details_data["nomCommune"] == 'DOUVRES')].groupby(["codeInsee","nomCommune","nomPeril","dateArrete", "franchise"]).size()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _BAR CHART ALL LIBELLE AVIS_
    """)
    return


@app.cell
def _(bar_distribution, details_data, mo):
    mo.mpl.interactive(bar_distribution(details_data, filter_year=1983))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART "NON RECONNUE"_
    """)
    return


@app.cell
def _(details_data, line_chart_global, mo):
    mo.mpl.interactive(line_chart_global(details_data, group_freq="3YE"))
    return


@app.cell
def _(details_data, line_chart_taux, mo):
    mo.mpl.interactive(line_chart_taux(df=details_data))
    return


@app.cell
def _(details_data, line_chart_taux, mo):
    mo.mpl.interactive(line_chart_taux(df=details_data, target_col="franchise"))
    return


@app.cell
def _(details_data, np, pd):
    details_par_catnat = details_data.copy()

    details_par_catnat["nomPeril"] = (
        details_par_catnat["nomPeril"]
        .mask(details_par_catnat["nomPeril"].str.contains("Inondations"), "Inondations")
        .where(details_par_catnat["nomPeril"].str.contains("Inondations|Sécheresse"), "Autre")
    )

    details_par_catnat = details_par_catnat[(details_par_catnat["dateArrete"].dt.year > 1983) & (details_par_catnat["dateArrete"].dt.year < 2026)]
    details_par_catnat['periode_2010'] = np.where(
        details_par_catnat['dateArrete'].dt.year < 2010, 
        'Avant 2010', 
        'Après 2010'
    )

    avis_par_catnat = (
        details_par_catnat.groupby(
            [pd.Grouper(key="dateArrete", freq="YE"), "nomPeril", "libelleAvis", "periode_2010"]
        )
        .size()
        .reset_index(name="count")
    )
    return avis_par_catnat, details_par_catnat


@app.cell
def _(details_par_catnat, donut_global_type_catnat, mo):
    mo.mpl.interactive(donut_global_type_catnat(details_par_catnat, font=None))
    return


@app.cell
def _(details_par_catnat, donut_specific, mo):
    mo.mpl.interactive(donut_specific(df=details_par_catnat, target_col="libelleAvis"))
    return


@app.cell
def _(avis_par_catnat, mo, stacked_area_type_catnat):
    mo.mpl.interactive(stacked_area_type_catnat(df=avis_par_catnat, window=5))
    return


@app.cell
def _(avis_par_catnat, bar_total_evolution, mo):
    mo.mpl.interactive(bar_total_evolution(df=avis_par_catnat))
    return


@app.cell
def _(avis_par_catnat, bar_catnat_evolution, mo):
    mo.mpl.interactive(bar_catnat_evolution(df=avis_par_catnat))
    return


@app.cell
def _(avis_par_catnat, fig_arrow, mo, mtick, np, plt, sns, source_text):
    def taux_catnat_ts(df, 
                       target_col="libelleAvis"):

        if target_col == "libelleAvis":
            select_cols = ["Non reconnue", "Reconnue"]
            title="Taux de Non-Reconnaissance"
        else:
            select_cols = ["Doublée ou plus", "Simple"]
            title="Taux de Franchises 'Doublée ou plus'"

        df_plot = df[df["nomPeril"] != "Autre"].pivot_table(
            index=['dateArrete', 'nomPeril'], 
            columns=target_col, 
            values='count', 
            fill_value=0
        ).reset_index()

        df_plot['ratio'] = df_plot[select_cols[0]] / (df_plot[select_cols[0]] + df_plot[select_cols[1]])
        df_plot['ratio'] = df_plot['ratio']\
            .replace([np.inf, -np.inf], np.nan).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_plot, 
            x=df_plot['dateArrete'].dt.year, 
            y='ratio', 
            palette=["#496988", "#AA4A44"],
            marker='o',
            hue='nomPeril',
            linewidth=2.5,
            ms=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1,
            ax=ax
        )

        plt.title(title, 
                  fontsize=18, color="black")
        plt.ylabel("Taux", fontsize=15, 
                   color="black")
        plt.xlabel("")
        plt.grid(True, linestyle='--', alpha=0.7)

        ax.tick_params(labelsize=10)
        ax.yaxis.set_label_coords(-0.06, 0.55)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.0, 1)

        plt.legend(title="", 
                   frameon=False,
                   fontsize=12,
                   labelcolor="black")

        if target_col == "libelleAvis":
            fig_arrow(
                head_position=(0.94, 0.88),
                tail_position=(0.85, 0.85),
                width=1,
                radius=-0.1,
                color="black",
                fill_head=False,
                mutation_scale=0.75,
            )

            plt.text(
                0.875,
                0.86,
                f"{(df_plot["ratio"].max() * 100).round(1)} %",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=14,
                color="#D2042D",
                fontweight='bold'
            )

            plt.text(
                0.975,
                -0.125,
                source_text,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
            )

        plt.tight_layout()
        return ax

    mo.mpl.interactive(taux_catnat_ts(df=avis_par_catnat))
    return (taux_catnat_ts,)


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART EVOLUTION OF FRANCHISE_
    """)
    return


@app.cell
def _(bar_distribution, details_par_catnat, mo):
    mo.mpl.interactive(bar_distribution(df=details_par_catnat, color="#FFAA33", analysis="franchise"))
    return


@app.cell
def _(details_data, line_chart_global, mo):
    mo.mpl.interactive(line_chart_global(details_data,group_freq="3YE",target_col="franchise"))
    return


@app.cell
def _(details_par_catnat, donut_specific, mo):
    mo.mpl.interactive(donut_specific(df=details_par_catnat, target_col="franchise"))
    return


@app.cell
def _(bar_total_evolution, details_par_catnat, pd):
    franchise_par_catnat = details_par_catnat.copy()
    franchise_par_catnat = franchise_par_catnat[franchise_par_catnat["dateArrete"].dt.year > 1984]
    franchise_par_catnat = franchise_par_catnat[
        franchise_par_catnat["libelleAvis"] == "Reconnue"
    ]

    franchise_par_catnat['franchise'] = franchise_par_catnat["franchise"].replace(
                                                              {"Doublée":"Doublée ou plus",
                                                               "Triplée":"Doublée ou plus",
                                                               "Quadruplée":"Doublée ou plus"})

    franchise_par_catnat = (
        franchise_par_catnat.groupby(
            [pd.Grouper(key="dateArrete", freq="YE"), "nomPeril", "franchise", "periode_2010"]
        )
        .size()
        .reset_index(name="count")
    )

    bar_total_evolution(franchise_par_catnat,
                        title="Evolution Franchises",
                        filter_col="franchise", 
                        split_year=2010,
                        font=None)
    return (franchise_par_catnat,)


@app.cell
def _(bar_catnat_evolution, franchise_par_catnat, mo):
    mo.mpl.interactive(bar_catnat_evolution(df=franchise_par_catnat, target_col="franchise"))
    return


@app.cell
def _(franchise_par_catnat, mo, stacked_area_type_catnat):
    mo.mpl.interactive(stacked_area_type_catnat(df=franchise_par_catnat, target_col="franchise"))
    return


@app.cell
def _(franchise_par_catnat, mo, taux_catnat_ts):
    mo.mpl.interactive(taux_catnat_ts(df=franchise_par_catnat, target_col="franchise"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART LIBELLE AVIS "NON RECONNUE" PER COMMUNES (%)_
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CARTO
    """)
    return


@app.cell
def _(gpd):
    regions_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-avec-outre-mer.geojson"
    dept_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-avec-outre-mer.geojson"
    communes_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes-avec-outre-mer.geojson"
    #"https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes.geojson"

    regions_geo = gpd.read_file(regions_geojson_url)
    dept_geo = gpd.read_file(dept_geojson_url)
    communes_geo = gpd.read_file(communes_geojson_url)
    return communes_geo, dept_geo


@app.cell
def _(communes_geo, details_par_catnat, np, pd):
    # --- 1. Preparation of Source Data ---
    df_work = details_par_catnat.copy()

    # Categorize franchises (Simple vs Double+)
    # Note: We do this before filtering to ensure the column exists
    conditions = [
        df_work["franchise"] == "Simple",
        df_work["franchise"] == "Doublée",
        df_work["franchise"].isin(["Triplée", "Quadruplée"])
    ]
    choices = ["Simple", "Doublée", "Triplée+"]
    df_work['f_cat'] = np.select(conditions, choices, default="Autre")

    # Create a boolean helper for counting high penalties
    df_work['is_triple_plus'] = df_work['f_cat'] == "Triplée+"

    # --- 2. Calculate Non-Recognition Rates (Access to Rights) ---
    # Pivot: Rows=Commune, Cols=(Peril, Avis)
    pivot_avis = df_work.pivot_table(
        index="codeInsee",
        columns=["nomPeril", "libelleAvis"],
        values="dateArrete", # Any column to count
        aggfunc="count",
        fill_value=0
    )

    # --- 3. Calculate Franchise Penalty Rates (Cost of Rights) ---
    # CRITICAL: Filter only RECOGNIZED events first. 
    # You cannot have a franchise on a non-recognized event.
    df_recognized = df_work[df_work["libelleAvis"] == "Reconnue"]

    pivot_franchise = df_recognized.pivot_table(
        index="codeInsee",
        columns=["nomPeril", "f_cat"],
        values="dateArrete",
        aggfunc="count",
        fill_value=0
    )

    # --- 4. Metric Calculation Loop ---
    # We create a final dataframe to store results
    final_stats = pd.DataFrame(index=pivot_avis.index)

    # A. Global Stats

    total_non_rec = pivot_avis.xs('Non reconnue', level=1, axis=1, drop_level=False).sum(axis=1)
    total_events = pivot_avis.sum(axis=1)
    final_stats['taux_non_rec_global'] = (total_non_rec / total_events.replace(0, 1) * 100).fillna(0)
    final_stats['num_total'] = total_events 

    total_penalized = (
        pivot_franchise.xs('Doublée', level=1, axis=1, drop_level=False).sum(axis=1) + 
        pivot_franchise.xs('Triplée+', level=1, axis=1, drop_level=False).sum(axis=1)
    )
    total_rec_events = pivot_franchise.sum(axis=1)

    final_stats['taux_penalite_global'] = (total_penalized / total_rec_events.replace(0, 1) * 100).fillna(0)
    final_stats['num_reconnue_total'] = total_rec_events

    final_stats['num_triple_plus_global'] = pivot_franchise.xs('Triplée+', level=1, axis=1, drop_level=False).sum(axis=1)

    # B. Peril-Specific Stats
    perils = ['Inondations', 'Sécheresse', 'Autre']

    for peril in perils:
        # 1. Non-Recognition Rates
        try:
            n_rec = pivot_avis.get((peril, 'Non reconnue'), 0)
            n_ok = pivot_avis.get((peril, 'Reconnue'), 0)
            total_p = n_rec + n_ok

            final_stats[f'num_{peril.lower()}'] = total_p
            final_stats[f'taux_non_rec_{peril.lower()}'] = (n_rec / total_p * 100).fillna(0)
        except KeyError:
            final_stats[f'num_{peril.lower()}'] = 0
            final_stats[f'taux_non_rec_{peril.lower()}'] = 0

        # 2. Franchise Penalty Rates
        try:
            f_simple = pivot_franchise.get((peril, 'Simple'), 0)
            f_double = pivot_franchise.get((peril, 'Doublée'), 0)
            f_triple = pivot_franchise.get((peril, 'Triplée+'), 0)
        
            total_f = f_simple + f_double + f_triple
        
            # Rate for this peril
            final_stats[f'taux_penalite_{peril.lower()}'] = ((f_double + f_triple) / total_f * 100).fillna(0)
        
            # Specific count of Triplée+ for this peril
            final_stats[f'num_triple_plus_{peril.lower()}'] = f_triple
        
        except KeyError:
            final_stats[f'taux_penalite_{peril.lower()}'] = 0
            final_stats[f'num_triple_plus_{peril.lower()}'] = 0

    # --- 5. Final Merge with Geometry ---
    communes_geo_taux = communes_geo.merge(
        final_stats, 
        left_on="code", 
        right_index=True, # Using index because final_stats is indexed by codeInsee
        how="left"
    ).fillna(0)
    communes_geo_taux
    return (communes_geo_taux,)


@app.cell
def _(np, pd):
    def prepare_growth_analysis(df, split_year=2010, end_year=2025):
        df = df.copy()
    
        df['year'] = pd.to_datetime(df['dateArrete']).dt.year
        df['period'] = np.where(df['year'] < split_year, 'pre', 'recent')
    
        if 'f_cat' not in df.columns:
            conditions = [
                df["franchise"] == "Simple",
                df["franchise"].isin(["Doublée", "Triplée", "Quadruplée", "Doublée+"]) 
            ]
        
            choices = ["Simple", "Doublée+"]
        
            df['f_cat'] = np.select(conditions, choices, default="Inconnu")

        years_pre = split_year - 1982
        years_recent = end_year - split_year

        stats = df.groupby(['codeInsee', 'period']).agg(
            total_events=('libelleAvis', 'count'),
            refusals=('libelleAvis', lambda x: (x == "Non reconnue").sum()),
            recognitions=('libelleAvis', lambda x: (x == "Reconnue").sum()),
            doubled_penalties=('f_cat', lambda x: (x == "Doublée+").sum())
        ).unstack(fill_value=0)

        stats['freq_pre'] = stats[('total_events', 'pre')] / years_pre
        stats['freq_recent'] = stats[('total_events', 'recent')] / years_recent
        stats['freq_growth'] = ((stats['freq_recent'] - stats['freq_pre']) / (stats['freq_pre'] + 0.01)) * 100

        stats['taux_non_rec_pre'] = (stats[('refusals', 'pre')] / stats[('total_events', 'pre')].replace(0, 1)) * 100
        stats['taux_non_rec_recent'] = (stats[('refusals', 'recent')] / stats[('total_events', 'recent')].replace(0, 1)) * 100
        stats['taux_non_rec_growth'] = stats['taux_non_rec_recent'] - stats['taux_non_rec_pre']

        stats['taux_penalite_pre'] = (stats[('doubled_penalties', 'pre')] / stats[('recognitions', 'pre')].replace(0, 1)) * 100
        stats['taux_penalite_recent'] = (stats[('doubled_penalties', 'recent')] / stats[('recognitions', 'recent')].replace(0, 1)) * 100
        stats['taux_penalite_growth'] = stats['taux_penalite_recent'] - stats['taux_penalite_pre']

        stats.columns = [f"{col[0]}_{col[1]}"[:-1] if isinstance(col, tuple) else col for col in stats.columns]
    
        keep_cols = [
            "freq_pre", "freq_recent", "freq_growth",
            "taux_non_rec_pre", "taux_non_rec_recent", "taux_non_rec_growth",
            "taux_penalite_pre", "taux_penalite_recent", "taux_penalite_growth"
        ]
    
        return stats[keep_cols]
    return (prepare_growth_analysis,)


@app.cell
def _(details_par_catnat, prepare_growth_analysis):
    communes_geo_growth = prepare_growth_analysis(details_par_catnat)
    communes_geo_growth
    return (communes_geo_growth,)


@app.cell
def _(communes_geo_growth, communes_geo_taux):
    communes_geo_params = communes_geo_taux.merge(communes_geo_growth, left_on="code", right_on = "codeInsee", how="left")
    mask = (communes_geo_params["num_total"] > 2) & (communes_geo_params["num_reconnue_total"] > 1)
    communes_geo_params = communes_geo_params[mask]
    communes_geo_params
    return (communes_geo_params,)


@app.cell
def _(mo):
    mo.md(r"""
    ### DISTRIBUTION & CORRELATION
    """)
    return


@app.cell
def _(plt, sns):
    def dist_violin(df, col="taux_non_rec"):
        df_melted = df.melt(
            value_vars=[f"{col}_pre", f"{col}_recent"],
            var_name="Période",
            value_name="Taux"
        )
        df_melted["Période"] = df_melted["Période"].map({
            f"{col}_pre": "Avant 2010",
            f"{col}_recent": "Après 2010"
        })
    
        _, ax = plt.subplots(figsize=(10, 6))
    
        sns.violinplot(
            data=df_melted, 
            y="Taux", 
            hue="Période",
            split=True,       
            inner=None,        
            palette={"Avant 2010": "#496988", "Après 2010": "#e63946"},
            gap=.0,               
            ax=ax
        )

        if col == "taux_non_rec":
            title="Distribution Taux de Non Reconnaissance"
        else:
            title="Distribution Taux Penalité"
    
        plt.title(title, fontsize=18, pad=20, color="black")
        plt.xlabel("")
        plt.ylabel("Taux (%)", color="black")
    
        ax.set_xticks([])
        ax.set_ylim(0, 100)
        sns.despine(bottom=True)
    
        plt.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, labelcolor="black")
        plt.tight_layout()

        return ax
    return (dist_violin,)


@app.cell
def _(communes_geo_params, dist_violin):
    dist_violin(df=communes_geo_params, col="taux_non_rec")
    return


@app.cell
def _(communes_geo_params, dist_violin):
    dist_violin(df=communes_geo_params, col="taux_penalite")
    return


@app.cell
def _(communes_geo_params):
    tom_data = communes_geo_params[communes_geo_params["freq_growth"]>0].sort_values(by="taux_penalite_global", ascending=False).head(50)

    tom_data = tom_data[["code", "nom", 
                          "taux_penalité_global", "taux_non_rec_global",
                          "taux_penalité_inondations", "taux_non_rec_inondations",
                          "taux_penalité_sécheresse", "taux_non_rec_sécheresse",
                          "num_inondations", "num_sécheresse"]].reset_index(drop=True)
    tom_data
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### MONOVARIATE
    """)
    return


@app.cell
def _(communes_geo, communes_geo_params, gpd, mpatches, plt):
    def prepare_france_geometry(df):
        if df.crs != "EPSG:2154":
            df = df.to_crs(epsg=2154)

        df['dept_temp'] = df['code'].apply(lambda x: x[:3] if x.startswith('97') else x[:2])

        drom_configs = {
            '971': {'pos': (-150000, 6540000), 'scale': 1.3, 'name': "Guadeloupe"},
            '972': {'pos': (50000, 6540000),   'scale': 1.8, 'name': "Martinique"},
            '973': {'pos': (-150000, 6140000), 'scale': 0.25, 'name': "Guyane"},
            '974': {'pos': (-150000, 6340000), 'scale': 1, 'name': "La Réunion"},
            '976': {'pos': (50000, 6340000),   'scale': 2.5, 'name': "Mayotte"},
        }

        df_main = df[~df['dept_temp'].str.startswith('97')].copy()
        df_droms = []

        for code, cfg in drom_configs.items():
            drom = df[df['dept_temp'] == code].copy()
            if not drom.empty:
                centroid = drom.geometry.unary_union.centroid
                drom.geometry = drom.geometry.translate(xoff=-centroid.x, yoff=-centroid.y)
                drom.geometry = drom.geometry.scale(xfact=cfg['scale'], yfact=cfg['scale'], origin=(0,0))

                # Box Position
                drom.geometry = drom.geometry.translate(xoff=cfg['pos'][0], yoff=cfg['pos'][1])
                df_droms.append(drom)

        return gpd.pd.concat([df_main] + df_droms), drom_configs

    def monovariate_map(df, col="taux_non_rec_global", cmap="copper_r", 
                        title="Taux de non-reconnaissance par commune", 
                        missing_color="white", norm=None, 
                        legend_label="(%)",
                        subtitle=None, linewidth=0.1, colorbar=True):

        df_plot, drom_configs = prepare_france_geometry(df)

        fig, ax = plt.subplots(figsize=(14, 12))

        df_plot.plot(
            ax=ax,
            column=col,
            cmap=cmap,
            norm=norm,
            legend=colorbar,
            legend_kwds={"shrink": 0.4, "pad": -0.05},
            missing_kwds={"color": missing_color, "edgecolor": "black"},
            edgecolor="black",
            linewidth=linewidth, 
        )

        cbar_ax = ax.get_figure().get_axes()[-1]
        cbar_ax.set_ylabel(f" {legend_label}", color="black")

        if subtitle:
            ax.text(0.5, 0.99, subtitle, transform=ax.transAxes, 
                    fontsize=14, color='dimgray', ha='center', style='italic')

        # Box size (width, height) in meters
        box_w, box_h = 180000, 150000 

        for code, cfg in drom_configs.items():
            x, y = cfg['pos']
            rect = mpatches.Rectangle(
                (x - box_w/2, y - box_h/2), box_w, box_h,
                linewidth=0.5, edgecolor='gray', facecolor='none', 
                linestyle='--', alpha=0.6
            )
            ax.add_patch(rect)

            # Add label above each box
            ax.text(x, y + (box_h/2) + 12500, cfg['name'], 
                    fontsize=9, ha='center', color='dimgray')

        ax.set_axis_off()
        ax.set_title(title, fontsize=22, fontweight='bold', pad=15, color="black")

        null_patch = mpatches.Patch(color=missing_color, label="Pas de cas", edgecolor="black")
        leg = ax.legend(handles=[null_patch], loc="lower left", frameon=False, bbox_to_anchor=(0.5, 0.025))

        plt.setp(leg.get_texts(), color='black', fontsize=10) 

        ax.set_xlim(-280000, 1250000)
        ax.set_ylim(6000000, 7150000)

        return fig

    missing_color = "#DBDBDB"
    communes_geo_plot = communes_geo.merge(communes_geo_params, how="left")
    monovariate_map(communes_geo_plot, missing_color=missing_color)
    return (
        communes_geo_plot,
        missing_color,
        monovariate_map,
        prepare_france_geometry,
    )


@app.cell
def _(communes_geo_plot, monovariate_map, sns):
    monovariate_map(communes_geo_plot, 
                    col="taux_non_rec_inondations", 
                    cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                    title="Taux de non-reconnaissance par commune",
                    subtitle="(Inondations)",
                    missing_color='missing_color')
    return


@app.cell
def _(communes_geo_plot, missing_color, monovariate_map):
    monovariate_map(communes_geo_plot, 
                    col="taux_non_rec_sécheresse", 
                    cmap="YlOrRd",
                    title="Taux de non-reconnaissance par commune",
                    subtitle="(Sécheresse)",
                    missing_color=missing_color)
    return


@app.cell
def _(communes_geo_plot, mcolors, missing_color, monovariate_map):
    vmin, vmax, vcenter = 0, 500, 100
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    monovariate_map(communes_geo_plot, 
                    col="freq_growth", 
                    cmap="Reds",
                    norm=norm,
                    title="Augmentation des Arrêtés", 
                    missing_color=missing_color,
                    subtitle="Avant/Après 2010")
    return


@app.cell
def _(communes_geo_plot, mcolors, missing_color, monovariate_map):
    norm2 = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)

    monovariate_map(communes_geo_plot, 
                    col="taux_non_rec_growth", 
                    cmap="Oranges",
                    norm=norm2,
                    title="Augmentation du Taux de Non-reconnaissance", 
                    missing_color=missing_color,
                    subtitle="Avant/Après 2010")
    return


@app.cell
def _(communes_geo_plot, load_cmap, mcolors, missing_color, monovariate_map):
    norm3 = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=50)

    monovariate_map(communes_geo_plot, 
                    col="taux_penalité_global", 
                    cmap=load_cmap("Exter"),
                    norm=norm3,
                    title="Taux de Penalité", 
                    subtitle="(Franchise Doublée ou +)",
                    missing_color=missing_color)
    return


@app.cell
def _(communes_geo_plot, load_cmap, mcolors, missing_color, monovariate_map):
    norm4 = mcolors.TwoSlopeNorm(vmin=0, vcenter=25, vmax=50)

    monovariate_map(communes_geo_plot, 
                    col="taux_penalite_growth", 
                    cmap=load_cmap("X87", reverse=True),
                    norm=norm4,
                    title="Augmentation du Taux Penalité", 
                    missing_color=missing_color,
                    subtitle="Avant/Après 2010")
    return


@app.cell
def _(communes_geo_plot, np, pd):
    # Change bivariate
    df_quant = communes_geo_plot.copy()

    quant = 3
    labels = np.arange(1,quant+1)
    cols = ["freq_growth", "taux_non_rec_growth", "taux_penalite_growth", "num_triple_plus_global"]
    for col in cols:
        df_quant[col] = (df_quant[col] -  df_quant[col].min())/(df_quant[col].max() -  df_quant[col].min())
        df_quant[f"{col}_class"] = pd.qcut(
                df_quant[col].rank(method='first'), 
                [0, 0.25, 0.5, 0.85], # 85%
                labels=labels
            )

    triple_high_mask = (
        (df_quant["freq_growth_class"] == quant) & 
        (df_quant["taux_non_rec_growth_class"] == quant) & 
        (df_quant["num_triple_plus_global_class"] == quant)
        #(df_quant["taux_penalite_growth_class"] == quant)
    )

    critical_communes = df_quant[triple_high_mask]
    critical_communes = critical_communes.groupby(["code"]).size()
    critical_communes = critical_communes.reset_index()
    #critical_communes = gpd.GeoDataFrame(critical_communes)
    return (critical_communes,)


@app.cell
def _(
    LinearSegmentedColormap,
    communes_geo,
    critical_communes,
    missing_color,
    monovariate_map,
):
    monovariate_map(communes_geo.merge(critical_communes, on="code", how="left"), 
                    col=0, 
                    cmap=LinearSegmentedColormap.from_list("monochrome", ["#880808", "#880808"]),
                    title="Communes critiques", 
                    subtitle="Forte croissance CatNat, Non Reconnaissance et Penalité",
                    missing_color=missing_color,
                    colorbar=False)
    return


@app.cell
def _(
    communes_geo,
    communes_geo_plot,
    mcolors,
    missing_color,
    monovariate_map,
    sns,
):
    norm5 = mcolors.TwoSlopeNorm(vmin=1, vcenter=4, vmax=7)
    map_df = communes_geo.merge(communes_geo_plot[communes_geo_plot["num_triple_plus_global"]>0].drop(columns='geometry'), on="code", how="left")
    monovariate_map(map_df, 
                    col="num_triple_plus_global", 
                    cmap=sns.color_palette("dark:salmon_r", as_cmap=True),
                    norm=norm5,
                    title="Franchises Triplée ou Quadruple", 
                    legend_label="",
                    missing_color=missing_color)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### BIVARIATE
    """)
    return


@app.cell
def _(communes_geo_params, plt, sns):
    def check_percentile(df, 
                         quant=[0.7, 0.9],
                         col="taux_non_rec_global",
                         title="Distribution cumulative du taux de non-reconnaissance",
                         xlabel="Taux (%)"):
        percentiles = df[col].quantile(quant)
        p_low = percentiles[quant[0]]
        p_high = percentiles[quant[1]]

        _, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(
            data=df,
            x=col,
            element="step",
            fill=False,
            cumulative=True,
            stat="density",
            common_norm=False,
            color="black",
            ax=ax,
        )

        ax.axvline(p_high, color="red", linestyle="--", label=f"{quant[1]}% ({p_high:.1f}%)")
        ax.axvline(p_low, color="orange", linestyle="--", label=f"{quant[0]}% ({p_low:.1f}%)")

        ax.axhline(0.7, color="gray", alpha=0.3, linestyle=":")
        ax.axhline(0.9, color="gray", alpha=0.3, linestyle=":")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probabilité cumulative")
        ax.legend()

        return percentiles.index, ax

    percentiles_tg, ax_per_tg = check_percentile(communes_geo_params)
    ax_per_tg
    return (check_percentile,)


@app.cell
def _(check_percentile, communes_geo_params):
    percentiles_nc, ax_per_nc  = check_percentile(communes_geo_params, quant=[0.5, 0.85],
                                                 col="num_total",
                                                 title="Distribution cumulative du nombre de CatNat",
                                                 xlabel="Nombre")
    ax_per_nc
    return


@app.cell
def _(
    communes_geo_params,
    font,
    matplotlib,
    mpatches,
    np,
    pd,
    plt,
    prepare_france_geometry,
):
    def bivariate_map(df, 
                      cols=["taux_global", "num_total"],
                      quants=[[0.7, 0.9],[0.5, 0.85]], # Not in use => with pd.qcut [0, 0.33, 0.66]
                      title="Bivariée: Nombre CatNat et Non Reconnaissance",
                      legend_labels=["Non-reconnaissance", "N CatNat"],
                      colors=[
                            "#e8e8e8",
                            "#b0d5df",
                            "#64acbe",
                            "#e4acac",
                            "#ad9ea5",
                            "#627f8c",
                            "#c85a5a",
                            "#985356",
                            "#574249",
                        ],
                        subtitle=None,
                        focus=False):

        df_plot = df.copy()
        df_plot, drom_configs = prepare_france_geometry(df_plot)

        # Norm & Quantile (Not in use) [Modify this function]
        #percentiles = []
        for col, quant in zip(cols, quants):
            df_plot[col] = (df_plot[col] -  df_plot[col].min())/(df_plot[col].max() -  df_plot[col].min())
            #perc = df_plot[col].quantile(quant)
            #percentiles.extend(perc.values)

        #bins_var1 = [0, percentiles[0], percentiles[1], 1]
        #bins_var2 = [0, percentiles[2], percentiles[3], 1]

        df_plot["Var1_Class"] = pd.qcut(
            df_plot[cols[0]].rank(method='first'),
            3
        )
        df_plot["Var1_Class"] = df_plot["Var1_Class"].astype("str")

        df_plot["Var2_Class"] = pd.qcut(
            df_plot[cols[1]].rank(method='first'), 
            3
        )
    
        df_plot["Var2_Class"] = df_plot["Var2_Class"].astype("str")

        x_class_codes = np.arange(1, 4)
        d = dict(
            zip(
                df_plot["Var1_Class"].value_counts().sort_index().index,
                x_class_codes,
            )
        )
        df_plot["Var1_Class"] = df_plot["Var1_Class"].replace(d)

        # Code created y bins to A, B, C
        y_class_codes = ["A", "B", "C"]
        d = dict(
            zip(
                df_plot["Var2_Class"].value_counts().sort_index().index,
                y_class_codes,
            )
        )
        df_plot["Var2_Class"] = df_plot["Var2_Class"].replace(d)

        # Combine x and y codes to create Bi_Class
        df_plot["Bi_Class"] = (
            df_plot["Var1_Class"].astype("str") + df_plot["Var2_Class"]
        )

        cmap = matplotlib.colors.ListedColormap(colors)

        # MAIN PLOT
        fig, ax = plt.subplots(figsize=(8, 8))

        df_plot.plot(
            ax=ax,
            column="Bi_Class", 
            cmap=cmap, 
            alpha=0.5 if focus else 1,
            categorical=True, 
            legend=False,
            linewidth=0.1
        )

        plt.tight_layout()
        plt.axis("off")
        ax.set_title(
            title,
            font=font,
            fontsize=20,
            color="black",
        )

        # Box size (width, height) in meters
        box_w, box_h = 180000, 150000 

        for code, cfg in drom_configs.items():
            x, y = cfg['pos']
            rect = mpatches.Rectangle(
                (x - box_w/2, y - box_h/2), box_w, box_h,
                linewidth=0.5, edgecolor='gray', facecolor='none', 
                linestyle='--', alpha=0.6
            )
            ax.add_patch(rect)

            # Add label above each box
            ax.text(x, y + (box_h/2) + 12500, cfg['name'], 
                    fontsize=9, ha='center', color='dimgray')

        # FOCUS
        if focus:
            darkest_label = "3C"
            high_high = df_plot[df_plot["Bi_Class"] == darkest_label]

            high_high.plot(
                ax=ax,
                color=colors[8],  # The darkest color (top-right of legend)
                edgecolor="black",
                linewidth=0.1,
                alpha=1.0,
            )

        img2 = fig
        ax2 = fig.add_axes([0.1, 0.675, 0.1, 0.1])

        alpha = 1

        # Column 1
        ax2.axvspan(
            xmin=0, xmax=0.33, ymin=0, ymax=0.33, alpha=alpha, color=colors[0]
        )
        ax2.axvspan(
            xmin=0, xmax=0.33, ymin=0.33, ymax=0.66, alpha=alpha, color=colors[1]
        )
        ax2.axvspan(
            xmin=0, xmax=0.33, ymin=0.66, ymax=1, alpha=alpha, color=colors[2]
        )

        # Column 2
        ax2.axvspan(
            xmin=0.33, xmax=0.66, ymin=0, ymax=0.33, alpha=alpha, color=colors[3]
        )
        ax2.axvspan(
            xmin=0.33, xmax=0.66, ymin=0.33, ymax=0.66, alpha=alpha, color=colors[4]
        )
        ax2.axvspan(
            xmin=0.33, xmax=0.66, ymin=0.66, ymax=1, alpha=alpha, color=colors[5]
        )

        # Column 3
        ax2.axvspan(
            xmin=0.66, xmax=1, ymin=0, ymax=0.33, alpha=alpha, color=colors[6]
        )
        ax2.axvspan(
            xmin=0.66, xmax=1, ymin=0.33, ymax=0.66, alpha=alpha, color=colors[7]
        )
        ax2.axvspan(
            xmin=0.66, xmax=1, ymin=0.66, ymax=1, alpha=alpha, color=colors[8]
        )

        # Annoate the legend
        ax2.tick_params(
            axis="both", which="both", length=0
        )  # remove ticks from the big box
        ax2.axis("off")  # turn off its axis
        ax2.annotate(
            "", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
        )  # draw arrow for x
        ax2.annotate(
            "", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
        )  # draw arrow for y
        ax2.text(
            s=legend_labels[0], x=0.1, y=-0.25, color="black", fontsize=9
        )  # annotate x axis
        ax2.text(
            s=legend_labels[1], x=-0.25, y=0.1, rotation=90, color="black", fontsize=9
        )  # annotate y axis

        # Highlight indicator on the legend 
        if focus:
            ax2.add_patch(
                plt.Rectangle((0.66, 0.66), 0.34, 0.34, fill=False, edgecolor="black", lw=1)
                )
        
        if subtitle:
            ax.text(0.5, 0.99, subtitle, transform=ax.transAxes, 
                    fontsize=14, color='dimgray', ha='center', style='italic')

        return fig

    bivariate_map(df=communes_geo_params,
                  cols=["taux_non_rec_global", "num_total"],
                  focus=False)
    return (bivariate_map,)


@app.cell
def _(bivariate_map, communes_geo_params):
    bivariate_map(df=communes_geo_params,
                  cols=["taux_non_rec_global", "num_total"],
                  focus=True)
    return


@app.cell
def _(bivariate_map, communes_geo_params):
    colors_ = [
                "#e8e8e8", 
                "#b0d5df", 
                "#64acbe",
                "#f9d89a", 
                "#c6bc9c", 
                "#8b9a9d", 
                "#f9b62c", 
                "#c0824b", 
                "#434c44", 
            ]

    bivariate_map(df=communes_geo_params,
                  cols=["num_sécheresse", "num_inondations"],
                  title="Double Peine: Nombre de Sécheresses et Inondations",
                  legend_labels=["Sécheresses", "Inondations"],
                  colors=colors_,
                  focus=False)
    return (colors_,)


@app.cell
def _(bivariate_map, colors_, communes_geo_params):
    bivariate_map(df=communes_geo_params,
                  cols=["num_sécheresse", "num_inondations"],
                  title="Double Peine: Nombre de Sécheresses et Inondations",
                  legend_labels=["Sécheresses", "Inondations"],
                  colors=colors_,
                  focus=True)
    return


@app.cell
def _(bivariate_map, communes_geo_params):
    colors_purple_orange = [
        "#e8e8e8", "#d3d3e8", "#6c6cb0", 
        "#f9d89a", "#c6bc9c", "#8b9a9d", 
        "#f9b62c", "#b05a5a", "#5a2a5a"  
    ]

    bivariate_map(df=communes_geo_params,
                  cols=["taux_non_rec_growth", "freq_growth"],
                  title="Double Peine: Augmentation des CatNat et Non Reconnaissance",
                  legend_labels=["Non reconnaissance", "Num Catnat"],
                  colors=colors_purple_orange,
                  subtitle="(Avant/Après 2010)",
                  focus=False)
    return (colors_purple_orange,)


@app.cell
def _(bivariate_map, colors_purple_orange, communes_geo_params):
    bivariate_map(df=communes_geo_params,
                  cols=["taux_non_rec_growth", "freq_growth"],
                  title="Double Peine: Augmentation des CatNat et Non Reconnaissance",
                  legend_labels=["Non reconnaissance", "Num Catnat"],
                  colors=colors_purple_orange,
                  subtitle="(Avant/Après 2010)",
                  focus=True)
    return


@app.cell
def _(bivariate_map, communes_geo_params):
    colors_vibrant_ob = [
        "#e8e8e8", "#b8d6be", "#73ae80", # Low Amber (Gray -> Pale Green -> Teal)
        "#f5d44f", "#b8a57d", "#5a9178", # Mid Amber (Yellow -> Olive -> Dark Teal)
        "#edad08", "#a3863d", "#2a5a5b"  # High Amber (Amber -> Brown -> Deep Ocean)
    ]

    bivariate_map(df=communes_geo_params,
                  cols=["taux_penalité_global", "taux_non_rec_global"],
                  title="Double Peine: Non reconnaissance et Penalité",
                  legend_labels=["Taux Penalité", "Non reconnaissance"],
                  colors=colors_vibrant_ob,
                  focus=False)
    return (colors_vibrant_ob,)


@app.cell
def _(bivariate_map, colors_vibrant_ob, communes_geo_params):
    bivariate_map(df=communes_geo_params,
                  cols=["taux_penalité_global", "taux_non_rec_global"],
                  title="Double Peine: Non reconnaissance et Penalité",
                  legend_labels=["Taux Penalité", "Non reconnaissance"],
                  colors=colors_vibrant_ob,
                  focus=True)
    return


@app.cell
def _():
    ## HIGH - HIGH - HIGH
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DEPT & REG ANALYSIS AND CARTO
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### DEPT TS ANALYSIS
    """)
    return


@app.cell
def _():
    # Mapping of Department Codes to Region Names
    dep_to_reg = {
        '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
        '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes', 
        '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes', 
        '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
        '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France', 
        '62': 'Hauts-de-France', '80': 'Hauts-de-France',
        '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur', 
        '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur', 
        '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
        '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est', 
        '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est', 
        '68': 'Grand Est', '88': 'Grand Est',
        '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie', 
        '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie', 
        '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
        '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
        '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine', 
        '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine', 
        '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine', 
        '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
        '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire', 
        '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
        '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté', 
        '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté', 
        '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
        '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
        '2A': 'Corse', '2B': 'Corse',
        '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire', 
        '72': 'Pays de la Loire', '85': 'Pays de la Loire',
        '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France', 
        '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
        '971': 'Guadeloupe', '972': 'Martinique', '973': 'Guyane', '974': 'La Réunion', '976': 'Mayotte'
    }

    def extract_dept(code):
        code = str(code).zfill(5)
        if code.startswith('97'):
            return code[:3] 
        return code[:2]   
    return dep_to_reg, extract_dept


@app.cell
def _(dep_to_reg, dept_geo, details_data, extract_dept, np, pd):
    def prepare_data_ts(df, window=5):
        df_ts = df.copy()
        df_ts['dateArrete'] = pd.to_datetime(df_ts['dateArrete'])
        df_ts['year'] = df_ts['dateArrete'].dt.year
        df_ts['window'] = (df_ts['year'] // window) * window

        df_ts["nomPeril"] = (
        df_ts["nomPeril"]
                .mask(df_ts["nomPeril"].str.contains("Inondations"), "Inondations")
                .where(df_ts["nomPeril"].str.contains("Inondations|Sécheresse"), "Autre")
            )

        df_ts['code_dept'] = df_ts['codeInsee'].apply(extract_dept)
    
        conditions = [
            df_ts["franchise"] == "Simple",
            df_ts["franchise"] == "Doublée",
            df_ts["franchise"].isin(["Triplée", "Quadruplée"])
        ]
        choices = ["Simple", "Doublée", "Triplée+"]
        df_ts['f_cat'] = np.select(conditions, choices, default="Autre")
    
        dept_stats = df_ts.groupby(['code_dept', 'window']).agg(
            num_total=('libelleAvis', 'count'),
            num_refus=('libelleAvis', lambda x: (x == "Non reconnue").sum()),
            num_ok=('libelleAvis', lambda x: (x == "Reconnue").sum())
        )
    
        franchise_counts = df_ts[df_ts["libelleAvis"] == "Reconnue"].groupby(['code_dept', 'window', 'f_cat']).size().unstack(fill_value=0)
    
        res = dept_stats.join(franchise_counts, how='left').fillna(0)
        res['taux_non_rec'] = (res['num_refus'] / res['num_total'] * 100)
    
        total_rec_with_data = res.get('Simple', 0) + res.get('Doublée', 0) + res.get('Triplée+', 0)
        res['taux_penalite'] = ((res.get('Doublée', 0) + res.get('Triplée+', 0)) / total_rec_with_data.replace(0, 1) * 100)
        res['num_triple_plus'] = res.get('Triplée+', 0)

        perils = ['Inondations', 'Sécheresse']
        for peril in perils:
            p_df = df_ts[df_ts['nomPeril'] == peril]
            p_stats = p_df.groupby(['code_dept', 'window']).agg(
                total=(f'libelleAvis', 'count'),
                refus=(f'libelleAvis', lambda x: (x == "Non reconnue").sum()),
                ok=(f'libelleAvis', lambda x: (x == "Reconnue").sum())
            )
            res[f'taux_non_rec_{peril.lower()}'] = (p_stats['refus'] / p_stats['total'] * 100).fillna(0)
            res[f'num_{peril.lower()}'] = p_stats['total'].reindex(res.index).fillna(0)
        
            p_rec = p_df[p_df["libelleAvis"] == "Reconnue"]
            if not p_rec.empty:
                p_f = p_rec.groupby(['code_dept', 'window', 'f_cat']).size().unstack(fill_value=0)
                p_total_f = p_f.sum(axis=1)
                res[f'taux_penalite_{peril.lower()}'] = ((p_f.get('Doublée', 0) + p_f.get('Triplée+', 0)) / p_total_f * 100).fillna(0)

        res['part_sécheresse'] = (res['num_sécheresse'] / res['num_total'].replace(0, 1) * 100).fillna(0)
        return res.reset_index()

    final_dept_stats = prepare_data_ts(details_data, window=1)
    dept_geo_ts = dept_geo.merge(final_dept_stats, left_on="code", right_on="code_dept")
    dept_geo_ts['region'] = dept_geo_ts['code_dept'].map(dep_to_reg)


    region_to_zone = {
        'Hauts-de-France': 'Nord', 'Normandie': 'Nord', 'Île-de-France': 'Nord',
        'Bretagne': 'Ouest', 'Pays de la Loire': 'Ouest', 'Nouvelle-Aquitaine': 'Ouest',
        'Grand Est': 'Est', 'Bourgogne-Franche-Comté': 'Est',
        'Auvergne-Rhône-Alpes': 'Centre/Est', 'Centre-Val de Loire': 'Centre',
        'Occitanie': 'Sud', 'Provence-Alpes-Côte d\'Azur': 'Sud', 'Corse': 'Sud',
        'Guadeloupe': 'DROM', 'Martinique': 'DROM', 'Guyane': 'DROM', 
        'La Réunion': 'DROM', 'Mayotte': 'DROM'
    }

    dept_geo_ts['zone'] = dept_geo_ts['region'].map(region_to_zone)
    dept_geo_ts = dept_geo_ts.sort_values("window")
    dept_geo_ts
    return (dept_geo_ts,)


@app.cell
def _(strip_data):
    strip_data
    return


@app.cell
def _(dept_geo_ts, pd, plt, sns):
    strip_data = dept_geo_ts[(dept_geo_ts["num_total"] > 1) & (dept_geo_ts["window"] >= 2005)].groupby(['nom', "zone"]).agg({
        'taux_penalite': 'mean',
        'taux_non_rec': 'mean',
        'num_total': 'sum'
    }).reset_index()


    strip_data_sorted = strip_data.sort_values("taux_non_rec")

    custom_order = ['Nord', 'Est', 'Ouest', 'Sud', 'Centre', 'Centre/Est', 'DROM']
    strip_data['zone'] = pd.Categorical(strip_data['zone'], categories=custom_order, ordered=True)

    g = sns.catplot( #scatterplot
        data=strip_data_sorted,
        x="taux_non_rec",
        y="zone",
        size=12,
        alpha=1,
        color="#F0E68C",
        height=6, 
        aspect=1.6,
        kind="strip",
        jitter=0.15,
        order=custom_order,
        edgecolor="black",    
        linewidth=.75,
    )

    ax = g.ax

    ax.margins(y=0)

    for ii in range(len(custom_order)):
        if ii % 2 == 0:
            ax.axhspan(ii - 0.5, ii + 0.5, color='#A7C7E7', zorder=0, alpha=0.3)
        else:
            ax.axhspan(ii - 0.5, ii + 0.5, color='#A7C7E7', zorder=0, alpha=0.6)

    # Labels and Title
    plt.title("Taux de Non-Reconnaissance par Zone et Départements", 
              fontsize=20, loc='left', pad=20, color="black")
    plt.xlabel("Taux (%)", fontsize=12)
    plt.ylabel("") 

    # --- GRID CONTROL ---
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)                                        
    ax.set_axisbelow(True)                                      

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()
    return custom_order, strip_data, strip_data_sorted


@app.cell
def _(custom_order, strip_data_sorted):
    import plotly.express as px
    import plotly.graph_objects as go


    # 2. Create the base strip plot
    # We use px.strip to maintain the "jitter" look
    fig_pl = px.strip(
        strip_data_sorted,
        x="taux_non_rec",
        y="zone",
        hover_name="nom", # This shows the Department name on hover
        custom_data=["num_total", "taux_penalite"],
        category_orders={"zone": custom_order}, # Maintains your Nord -> DROM order
        title="Taux de Non-Reconnaissance par Zone et Départements",
    )

    # 3. Styling the markers (Matching your #F0E68C and black edges)
    fig_pl.update_traces(
        marker=dict(
            size=12,
            color='#F0E68C',
            line=dict(width=0.75, color='black')
        ),
        jitter=0.3 # Adjusts the vertical spread
    )

    # 4. Adding the "Zebra" Background Spans
    for j, zone in enumerate(custom_order):
        # Set alternating opacity to match your logic
        opacity = 0.3 if j % 2 == 0 else 0.6
    
        fig_pl.add_hrect(
            y0=j - 0.5, y1=j + 0.5,
            fillcolor="#A7C7E7",
            opacity=opacity,
            layer="below",
            line_width=0,
        )

    # 5. Refining Layout and Tooltips
    fig_pl.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(title="Taux (%)", showgrid=False, zeroline=False),
        yaxis=dict(title="", showgrid=False),
        font=dict(family="Arial", size=12),
        title_font=dict(size=24),
        margin=dict(l=100, r=20, t=80, b=40),
        hoverlabel=dict(bgcolor="white", font_size=13)
    )

    # Customize what appears in the tooltip
    fig_pl.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Taux Non-Rec: %{x:.1f}%<br>Total Événements: %{customdata[0]}<br>Taux Pénalité: %{customdata[1]:.1f}%<extra></extra>"
    )

    fig_pl.write_html("catnat_analysis.html", include_plotlyjs='cdn')
    fig_pl.show()
    return


@app.cell
def _(custom_order, dept_geo_ts, plt, sns):
    plt.figure(figsize=(10, 6))

    df_plot = dept_geo_ts[(dept_geo_ts["num_total"] > 1) & (dept_geo_ts["window"] >= 2005)].groupby(['nom', "zone"]).agg({
                                                                'taux_penalite': 'mean',
                                                                'taux_non_rec': 'mean',
                                                                'part_sécheresse': 'mean',
                                                                'num_total': 'sum'
                                                            }).reset_index()

    ax_test = sns.regplot(
        data=df_plot, 
        x="taux_non_rec", 
        y="part_sécheresse", 
        scatter=False,           # Don't draw points again
        color="black",           # Neutral color for the trend
        line_kws={"linestyle": "--", "label": "Tendance globale", "linewidth":1.75},
        ci=None,          # <--- Removes the shaded area
    )


    sns.scatterplot(
        data=df_plot,
        x="taux_non_rec",
        y="part_sécheresse",
        hue="zone",  
        size="num_total", 
        sizes=(50, 800),
        hue_order=custom_order,
        palette="Spectral",    
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5
    )

    # --- THE SEPARATE & SPACED LEGEND LOGIC ---

    # Get all handles and labels from the current axis
    handles, labels_ = ax_test.get_legend_handles_labels()

    try:
        zone_idx = labels_.index('zone')
        size_idx = labels_.index('num_total')
    
        # Legend 1: Trend & Zones
        # We take the first handle (trend) + the items between 'zone' and 'num_total'
        l1 = ax_test.legend(
            handles[zone_idx+1:size_idx], 
            labels_[zone_idx+1:size_idx],
            title="Macro-Zones",
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            labelspacing=1.2,       
            borderpad=1,            
            frameon=False,
            markerscale=2.0
        )
        ax_test.add_artist(l1) 

        # Legend 2: Size (num_total)
        # We take everything after the 'num_total' label
        l2 = ax_test.legend(
            handles[size_idx+1:], 
            labels_[size_idx+1:],
            title="Nombre d'arrêtés",
            bbox_to_anchor=(1.05, 0.4), 
            loc='upper left',
            labelspacing=1.5,           
            borderpad=1,
            frameon=False,
            handletextpad=2            
        )

    except ValueError:
        ax_test.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=1.5)

    plt.xlim(-2, 105)
    plt.ylim(-2, 90)

    # Labels and Title
    plt.title("Correlation entre Sécheresse et Non-Reconnaissance", fontsize=16, color="black", fontweight="bold")
    plt.xlabel("Taux de Non-Reconnaissance (%)", fontsize=10)
    plt.ylabel("Part de Catnat Secheresse (%)", fontsize=10)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### DEPT CARTO
    """)
    return


@app.cell
def _(communes_geo_params, dept_geo, extract_dept):
    dept_geo_params = communes_geo_params.copy()
    dept_geo_params['code'] = dept_geo_params['code'].apply(extract_dept)
    dept_geo_params = dept_geo_params.drop(columns = ["geometry", "nom"])
    dept_geo_params = dept_geo_params.groupby("code").mean()
    dept_geo_params = dept_geo.merge(dept_geo_params, on="code")
    dept_geo_params
    return (dept_geo_params,)


@app.cell
def _(dept_geo_params, monovariate_map):
    monovariate_map(dept_geo_params,
                    title="Taux de non-reconnaissance par departement", 
                    missing_color="grey")
    return


@app.cell
def _(dept_geo_params, monovariate_map, sns):
    monovariate_map(dept_geo_params,
                    col="taux_non_rec_inondations",
                    title="Taux de non-reconnaissance par departement",
                    subtitle="(Inondations)",
                    cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                    missing_color="grey")
    return


@app.cell
def _(dept_geo_params, monovariate_map):
    monovariate_map(dept_geo_params,
                    col="taux_non_rec_sécheresse",
                    title="Taux de non-reconnaissance par departement",
                    subtitle="(Sécheresse)",
                    cmap="YlOrRd",
                    missing_color="grey")
    return


@app.cell
def _(bivariate_map, dept_geo_params):
    bivariate_map(df=dept_geo_params,
                  cols=["taux_non_rec_global", "num_total"],
                  focus=False)
    return


@app.cell
def _(bivariate_map, dept_geo_params):
    bivariate_map(df=dept_geo_params,
                  cols=["taux_non_rec_sécheresse", "num_sécheresse"],
                  focus=False)
    return


@app.cell
def _(bivariate_map, colors_, dept_geo_params):
    bivariate_map(df=dept_geo_params,
                  cols=["num_sécheresse", "num_inondations"],
                  title="Double Peine: Nombre de Sécheresses et Inondations",
                  legend_labels=["Sécheresses", "Inondations"],
                  colors=colors_,
                  focus=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CLUSTERING
    """)
    return


@app.cell
def _():
    """
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)


    pca = FastICA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. Create a DataFrame for plotting
    df_plot = communes_geo_params.copy()
    df_plot['cluster'] = kmeans.labels_
    df_plot['pca_1'] = X_pca[:, 0]
    df_plot['pca_2'] = X_pca[:, 1]

    # 3. Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='pca_1', y='pca_2', hue="cluster", palette='viridis', alpha=0.7)
    plt.title('K-Means Clusters (PCA-reduced 2D space)')
    plt.show()
    """
    return


@app.cell
def _():
    """
    scaler = QuantileTransformer()

    X = scaler.fit_transform(communes_geo_params[["num_sécheresse", 
                                                  "num_inondations", 
                                                  "num_total", 
                                                  "taux_inondations", 
                                                  "taux_sécheresse",
                                                  "taux_global"]])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(X)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    df_plot = communes_geo_params.copy()
    df_plot['umap_1'] = embedding[:, 0]
    df_plot['umap_2'] = embedding[:, 1]
    df_plot['cluster'] = cluster_labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='umap_1', y='umap_2', 
        hue='cluster', 
        data=df_plot,
        palette='viridis',
        alpha=0.6,
        edgecolor='w'
    )
    plt.title('HDBSCAN Clusters projected via UMAP', fontsize=15)
    plt.show()
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## GIF
    """)
    return


@app.cell
def _(
    communes_geo,
    details_data,
    imageio,
    mpatches,
    np,
    os,
    pd,
    plt,
    prepare_france_geometry,
):

    def create_catnat_franchise_gif(df, output_file="franchise_evolution.gif"):
        # 1. Clean Data and Define Penalty Rank
        df = df.copy()
        df['year'] = pd.to_datetime(df['dateArrete']).dt.year
    
        # Define the hierarchy for the "max" penalty logic
        penalty_mapping = {
            "Simple": 1,
            "Doublée": 2,
            "Triplée": 3,
            "Quadruplée": 4  
        }
        df['p_rank'] = df['franchise'].map(penalty_mapping).fillna(0)
    
        rank_to_label = {1: "Simple", 2: "Doublée", 3: "Triplée", 4: "Quadruplée"}
    
        color_dict = {
            "Simple": "#4F7942", 
            "Doublée": "#feb24c", 
            "Triplée": "#e31a1c",
            "Quadruplée": "#8B0000"
        }

        temp_dir = "temp_frames_franchise"
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        filenames = []

        for year in np.arange(2000,2026):
            # Get max penalty rank per commune for this year (Maybe change this)
            df_year = df[df['year'] == year]
            yearly_max = df_year.groupby('codeInsee')['p_rank'].max().reset_index()
            yearly_max['f_label'] = yearly_max['p_rank'].map(rank_to_label)
        
            df_plot = communes_geo.merge(yearly_max, left_on='code', right_on="codeInsee", how='left')
            df_plot, drom_configs = prepare_france_geometry(df_plot)

            fig, ax = plt.subplots(figsize=(12, 12))
        
            df_plot.plot(ax=ax, color='#B2BEB5', edgecolor='dimgrey', linewidth=0.05)
        
            active_zones = df_plot[df_plot['p_rank'] > 0].copy()
        
            if not active_zones.empty:
                active_zones['color_to_plot'] = active_zones['f_label'].map(color_dict)
            
                active_zones.plot(
                    ax=ax,
                    color=active_zones['color_to_plot'], # Pass the mapped column
                    legend=False, # We will handle the legend manually below
                    edgecolor='black',
                    linewidth=0.05
                )

            
                legend_patches = [
                    mpatches.Patch(color=color, label=label) 
                    for label, color in color_dict.items()
                ]
                ax.legend(handles=legend_patches, loc='center right', frameon=False, title="")

            # Handle DROM boxes (keeping your existing logic)
            box_w, box_h = 180000, 150000 
            for code, cfg in drom_configs.items():
                x, y = cfg['pos']
                rect = mpatches.Rectangle((x - box_w/2, y - box_h/2), box_w, box_h,
                                         linewidth=0.5, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.4)
                ax.add_patch(rect)
                ax.text(x, y + (box_h/2) + 12500, cfg['name'], fontsize=10, ha='center', color='dimgray')

            # Annotations
            ax.set_axis_off()
            ax.text(0.04, 0.92, f"Franchises: {year}", transform=ax.transAxes, 
                    fontsize=24, fontweight='bold', color='#333333')
        
            # Save frame
            filename = f"{temp_dir}/frame_{year}.png"
            plt.savefig(filename, dpi=120, bbox_inches='tight')
            plt.close()
            filenames.append(filename)

        # Stitch GIF
        print("Generating GIF...")
        with imageio.get_writer(output_file, mode='I', duration=500, loop=0) as writer:
            for filename in filenames:
                writer.append_data(imageio.imread(filename))
            
        print(f"Animation complete: {output_file}")

    create_catnat_franchise_gif(details_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
