import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="medium",
    layout_file="layouts/marimo_ccr_explore.slides.json",
)


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
def _():
    from pathlib import Path

    import altair as alt
    import contextily as ctx
    import geopandas as gpd
    import marimo as mo
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import morethemes as mt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from drawarrow import fig_arrow

    # Fonts, Palettes, Themes, Arrow
    from pyfonts import load_google_font
    from pypalettes import load_palette

    return (
        Path,
        alt,
        fig_arrow,
        gpd,
        load_google_font,
        matplotlib,
        mdates,
        mo,
        mpatches,
        mt,
        mtick,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(Path, load_google_font, mt):
    font = load_google_font("Roboto", italic=True)
    mt.set_theme("nature")

    current_dir = Path.cwd()
    seeds_dir = current_dir / "data" / "dbt_pipeline" / "seeds"
    return font, seeds_dir


@app.cell
def _(mo):
    mo.md(r"""
    ## _READING MAIN DATA_
    """)
    return


@app.cell
def _(pd, seeds_dir):
    main_data = pd.read_csv(seeds_dir / "ccr_main_page.csv")
    main_data.head(5)
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
def _(dropdown, main_data, pd):
    main_data["dateArrete"] = pd.to_datetime(main_data["dateArrete"], format="%d-%m-%Y")

    main_data_grouped = (
        main_data.set_index("dateArrete")
        .resample(dropdown.selected_key)
        .size()
        .reset_index(name="count")
        .set_index("dateArrete")
        .loc[:"2025"]
    )
    return (main_data_grouped,)


@app.cell
def _(dropdown, font, formats, main_data_grouped, mdates, mo, pd, plt):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(
        main_data_grouped.index,
        main_data_grouped["count"],
        linestyle="-",
        color="black",
    )

    plt.title("Evolution des Arrêtés", font=font, fontsize=20, color="black", pad=20)
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
    plt.xlim([main_data_grouped.index.min(), pd.to_datetime("2025-12-31")])

    source_text = "Source: CCR (Caisse Centrale de Réassurance)"
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
    mo.mpl.interactive(ax)
    return ax, source_text


@app.cell
def _(mo):
    mo.md(r"""
    ## _READING DETAILS DATA_
    """)
    return


@app.cell
def _(pd, seeds_dir):
    details_data = pd.read_csv(seeds_dir / "ccr_details.csv")
    details_data.head(5)
    return (details_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ### _BAR CHART ALL LIBELLE AVIS_
    """)
    return


@app.cell
def _(details_data):
    libelle_avis = details_data.copy()
    libelle_avis["libelleAvis"] = libelle_avis["libelleAvis"].replace(
        {
            "Reconnue(sans impact sur la modulation)": "Reconnue\n(sans impact sur la modulation)"
        }
    )

    libelle_avis = (
        libelle_avis.groupby("libelleAvis").count().sort_values(by="nomCommune")
    )
    return (libelle_avis,)


@app.cell
def _(ax, font, libelle_avis, mo, plt, source_text):

    _, ax1 = plt.subplots(figsize=(9, 5))
    bars = plt.barh(
        libelle_avis.index, libelle_avis["nomCommune"], color="#C4C4C4", height=0.55
    )

    plt.title("Avis des Arrêtés", font=font, fontsize=20, color="black", pad=20)
    plt.text(
        0.5,
        1.02,
        "(Par Communes)",
        font=font,
        fontsize=10,
        color="black",
        ha="center",
        transform=ax1.transAxes,
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

    percent_nr = (
        libelle_avis["nomCommune"].loc["Non reconnue"]
        / libelle_avis["nomCommune"].sum()
    )
    plt.text(
        0.65,
        0.45,
        f"Le pourcentage d'arrêtés $\mathbf{{Non\ Reconnue}}$ est de $\mathbf{{{(percent_nr * 100).round(1)}}}$%",
        font=font,
        fontsize=10,
        color="black",
        ha="center",
        transform=ax1.transAxes,
    )

    plt.text(
        0.85,
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

    mo.mpl.interactive(ax1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART INCREASE OF "NON RECONNUE"_
    """)
    return


@app.cell
def _(details_data, pd):
    libelle_avis_temp = details_data.copy()
    libelle_avis_temp["dateArrete"] = pd.to_datetime(libelle_avis_temp["dateArrete"])

    libelle_avis_temp["libelleAvis"] = libelle_avis_temp["libelleAvis"].replace(
        {
            "Reconnue(sans impact sur la modulation)": "Reconnue\n(sans impact sur la modulation)"
        }
    )

    libelle_avis_temp = (
        libelle_avis_temp.groupby(
            [pd.Grouper(key="dateArrete", freq="YE"), "libelleAvis"]
        )
        .size()
        .reset_index(name="count")
    )
    libelle_avis_temp = libelle_avis_temp[
        libelle_avis_temp["dateArrete"].dt.year <= 2026
    ]
    return (libelle_avis_temp,)


@app.cell
def _(font, libelle_avis_temp, mdates, mo, pd, plt, source_text):
    plot_data = libelle_avis_temp.pivot(
        index="dateArrete", columns="libelleAvis", values="count"
    ).fillna(0)
    plot_data = plot_data.loc[:"2025-12-31"]

    _, ax2 = plt.subplots(figsize=(10, 6))

    for column in plot_data.columns:
        ms = 5

        if "Non reconnue" in column:
            color = "#e63946"
            linewidth = 3
            zorder = 5

            ax2.plot(
                plot_data.index,
                plot_data[column],
                label=column,
                color=color,
                linewidth=linewidth,
                marker="o",
                linestyle="-",
                ms=ms,
                markerfacecolor="white",
                markeredgewidth=1,
                zorder=zorder,
            )
        else:
            ax2.plot(
                plot_data.index,
                plot_data[column],
                label=column,
                marker="o",
                linestyle="-",
                ms=ms,
                markerfacecolor="white",
                markeredgewidth=1,
                alpha=0.75,
                zorder=1,
            )

    plt.title("Evolution des Arrêtés", font=font, fontsize=20, color="black", pad=20)
    plt.text(
        0.5,
        1.01,
        "(Par type d'avis)",
        font=font,
        fontsize=11,
        color="black",
        ha="center",
        transform=ax2.transAxes,
    )

    plt.xlabel("")
    plt.ylabel("Occurrences", font=font, fontsize=14, color="black")
    ax2.yaxis.set_label_coords(-0.06, 0.65)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=30))

    plt.xticks(font=font, rotation=15, fontsize=9)
    plt.yticks(font=font, fontsize=10)

    plt.xlim([plot_data.index.min(), pd.to_datetime("2026-03-31")])

    plt.legend(prop=font, loc="upper right", frameon=False, labelcolor="black")

    plt.text(
        1,
        -0.125,
        source_text,
        ha="right",
        va="bottom",
        transform=ax2.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )

    plt.tight_layout()
    mo.mpl.interactive(ax2)
    return


@app.cell
def _(
    fig_arrow,
    font,
    libelle_avis_temp,
    mdates,
    mo,
    mtick,
    pd,
    plt,
    source_text,
):
    plot_data1 = libelle_avis_temp.pivot(
        index="dateArrete", columns="libelleAvis", values="count"
    ).fillna(0)
    plot_data1["Reconnue"] = (
        plot_data1["Reconnue"] + plot_data1["Reconnue\n(sans impact sur la modulation)"]
    )
    plot_data1 = plot_data1.drop(columns="Reconnue\n(sans impact sur la modulation)")
    plot_data1 = plot_data1["Non reconnue"] / (
        plot_data1["Reconnue"] + plot_data1["Non reconnue"]
    )
    plot_data1 = plot_data1.loc[:"2025-12-31"]

    _, ax2_1 = plt.subplots(figsize=(10, 6))

    color_ = "#e63946"
    linewidth_ = 3

    ax2_1.plot(
        plot_data1.index,
        plot_data1,
        color=color_,
        linewidth=linewidth_,
        marker="o",
        linestyle="-",
        ms=5,
        markerfacecolor="white",
        markeredgewidth=1,
    )

    plt.title(
        "Taux de Non Reconnaissance", font=font, fontsize=20, color="black", pad=20
    )

    plt.xlabel("")
    ax2_1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2_1.xaxis.set_major_locator(mdates.MonthLocator(interval=30))
    plt.xticks(font=font, rotation=15, fontsize=9)
    plt.xlim([plot_data1.index.min(), pd.to_datetime("2026-03-31")])

    plt.ylabel("Taux", font=font, fontsize=14, color="black")
    ax2_1.yaxis.set_label_coords(-0.06, 0.65)
    ax2_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    plt.yticks(font=font, fontsize=10)

    plt.legend(prop=font, loc="upper right", frameon=False, labelcolor="black")
    plt.text(
        1,
        -0.125,
        source_text,
        ha="right",
        va="bottom",
        transform=ax2_1.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )

    fig_arrow(
        head_position=(0.975, 0.86),
        tail_position=(0.88, 0.9),
        width=1,
        radius=0.1,
        color="black",
        fill_head=False,
        mutation_scale=0.75,
    )

    plt.text(
        0.875,
        0.99,
        f"Taux de {(plot_data1.max() * 100).round(2)}%",
        ha="right",
        va="bottom",
        transform=ax2_1.transAxes,
        fontsize=12,
        color="black",
        font=font,
    )

    fig_arrow(
        head_position=(0.65, 0.255),
        tail_position=(0.7, 0.2),
        width=1,
        radius=-0.1,
        color="black",
        fill_head=False,
        mutation_scale=0.75,
    )

    plt.text(
        0.805,
        0.075,
        f"Taux de {(plot_data1[plot_data1 > 0].min() * 100).round(2)}%",
        ha="right",
        va="bottom",
        transform=ax2_1.transAxes,
        fontsize=12,
        color="black",
        font=font,
    )

    plt.tight_layout()
    mo.mpl.interactive(ax2_1)
    return (plot_data1,)


@app.cell
def _(alt, mo, plot_data1):
    # Reset index so 'dateArrete' is a column for Altair
    data_for_altair = plot_data1.reset_index()
    data_for_altair.columns = ["Date", "Taux"]

    chart = (
        alt.Chart(data_for_altair)
        .mark_line(color="#e63946", strokeWidth=3, point=True)
        .encode(
            x=alt.X("Date:T", title="Année"),
            y=alt.Y("Taux:Q", title="Taux [%]", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d", title="Date de l'arrêté"),
                alt.Tooltip("Taux:Q", format=".2%", title="Taux de non-reconnue"),
            ],
        )
        .properties(width=900, height=400, title="Taux de Non Reconnaissance")
        .interactive()
    )

    # Display in marimo
    mo.ui.altair_chart(chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _BAR CHART ALL LIBELLE AVIS (AFTER 1983)_
    """)
    return


@app.cell
def _(mo):
    start_year = mo.ui.number(
        start=1982, stop=2025, step=1, value=1983, label="Année de début :"
    )
    start_year
    return (start_year,)


@app.cell
def _(details_data, pd, start_year):
    libelle_avis_a_1983 = details_data.copy()
    libelle_avis_a_1983["libelleAvis"] = libelle_avis_a_1983["libelleAvis"].replace(
        {
            "Reconnue(sans impact sur la modulation)": "Reconnue\n(sans impact sur la modulation)"
        }
    )

    libelle_avis_a_1983["dateArrete"] = pd.to_datetime(
        libelle_avis_a_1983["dateArrete"]
    )
    libelle_avis_a_1983 = libelle_avis_a_1983[
        libelle_avis_a_1983["dateArrete"].dt.year > start_year.value
    ]
    libelle_avis_a_1983 = (
        libelle_avis_a_1983.groupby("libelleAvis").count().sort_values(by="nomCommune")
    )
    return (libelle_avis_a_1983,)


@app.cell
def _(font, libelle_avis_a_1983, mo, plt, source_text, start_year):

    _, ax3 = plt.subplots(figsize=(9, 5))
    bars3 = plt.barh(
        libelle_avis_a_1983.index,
        libelle_avis_a_1983["nomCommune"],
        color="#C4C4C4",
        height=0.55,
    )

    plt.title("Avis des Arrêtés ", font=font, fontsize=20, color="black", pad=20)
    plt.text(
        0.5,
        1.02,
        f"(Par Communes après {start_year.value})",
        font=font,
        fontsize=10,
        color="black",
        ha="center",
        transform=ax3.transAxes,
    )  # subtitle

    plt.ylabel("")

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Add labels to each bar
    for bar3 in bars3:
        width3 = bar3.get_width()
        plt.gca().text(
            width3 + width3 * 0.025,
            bar3.get_y() + bar3.get_height() / 2.0,
            f"{int(width3):,}".replace(",", "'"),
            ha="left",
            va="center",
            color="black",
            font=font,
        )

    plt.yticks(font=font, fontsize=11)

    percent_nr_a_1983 = (
        libelle_avis_a_1983["nomCommune"].loc["Non reconnue"]
        / libelle_avis_a_1983["nomCommune"].sum()
    )
    plt.text(
        0.7,
        0.35,
        f"Le pourcentage d'arrêtés $\mathbf{{Non\ Reconnue}}$ est de $\mathbf{{{(percent_nr_a_1983 * 100).round(1)}}}$%",
        font=font,
        fontsize=10,
        color="black",
        ha="center",
        transform=ax3.transAxes,
    )

    plt.text(
        0.85,
        -0.075,
        source_text,
        ha="right",
        va="bottom",
        transform=ax3.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )
    plt.grid(False)
    plt.tight_layout()

    mo.mpl.interactive(ax3)
    return (ax3,)


@app.cell
def _(mo):
    mo.md(r"""
    ### BAR CHART TYPE DE FRANCHISE (AFTER 1983)
    """)
    return


@app.cell
def _(details_data, pd):
    franchise_a_1983 = details_data.copy()
    franchise_a_1983["dateArrete"] = pd.to_datetime(franchise_a_1983["dateArrete"])
    franchise_a_1983 = franchise_a_1983[
        franchise_a_1983["libelleAvis"] != "Non reconnue"
    ]
    franchise_a_1983 = franchise_a_1983[franchise_a_1983["dateArrete"].dt.year > 1983]
    franchise_a_1983 = (
        franchise_a_1983.groupby("franchise").count().sort_values(by="nomCommune")
    )
    return (franchise_a_1983,)


@app.cell
def _(font, franchise_a_1983, mo, plt, source_text):

    _, ax4 = plt.subplots(figsize=(9, 5))
    bars4 = plt.barh(
        franchise_a_1983.index,
        franchise_a_1983["nomCommune"],
        color="#73B373",
        height=0.55,
    )

    plt.title("Franchise des Arrêtés ", font=font, fontsize=20, color="black", pad=20)
    plt.text(
        0.5,
        1.02,
        "(Par Communes après 1983)",
        font=font,
        fontsize=10,
        color="black",
        ha="center",
        transform=ax4.transAxes,
    )  # subtitle

    plt.ylabel("")

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Add labels to each bar
    for bar4 in bars4:
        width4 = bar4.get_width()
        plt.gca().text(
            width4 + 1500,
            bar4.get_y() + bar4.get_height() / 2.0,
            f"{int(width4):,}".replace(",", "'"),
            ha="left",
            va="center",
            color="black",
            font=font,
        )

    plt.yticks(font=font, fontsize=11)

    plt.text(
        0.85,
        -0.075,
        source_text,
        ha="right",
        va="bottom",
        transform=ax4.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )
    plt.grid(False)
    plt.tight_layout()

    mo.mpl.interactive(ax4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART EVOLUTION OF FRANCHISE_
    """)
    return


@app.cell
def _(details_data, pd):
    franchise_temp = details_data.copy()
    franchise_temp["dateArrete"] = pd.to_datetime(franchise_temp["dateArrete"])
    franchise_temp = franchise_temp[franchise_temp["libelleAvis"] != "Non reconnue"]

    franchise_temp = (
        franchise_temp.groupby([pd.Grouper(key="dateArrete", freq="3YE"), "franchise"])
        .size()
        .reset_index(name="count")
    )
    franchise_temp = franchise_temp[franchise_temp["dateArrete"].dt.year <= 2026]
    return (franchise_temp,)


@app.cell
def _(font, franchise_temp, mdates, mo, plt, source_text):
    plot_data_1 = franchise_temp.pivot(
        index="dateArrete", columns="franchise", values="count"
    ).fillna(0)

    _, ax5 = plt.subplots(figsize=(10, 6))

    for column1 in ["Doublée", "Triplée", "Quadruplée"]:  # plot_data_1.columns:
        ax5.plot(
            plot_data_1.index,
            plot_data_1[column1],
            label=column1,
            marker="o",
            linestyle="-",
            ms=5,
            markerfacecolor="white",
            markeredgewidth=1,
            alpha=0.75,
            zorder=1,
        )

    plt.title("Evolution des Franchises", font=font, fontsize=20, color="black", pad=20)
    plt.text(
        0.5,
        1.01,
        "(Par type de franchise, exclusion Simple, Moyenne 3 Ans)",
        font=font,
        fontsize=11,
        color="black",
        ha="center",
        transform=ax5.transAxes,
    )

    plt.xlabel("")
    plt.ylabel("Occurrences", font=font, fontsize=14, color="black")
    ax5.yaxis.set_label_coords(-0.06, 0.65)

    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=30))

    plt.xticks(font=font, rotation=15, fontsize=9)
    plt.yticks(font=font, fontsize=10)

    plt.legend(prop=font, loc="upper left", frameon=False, labelcolor="black")

    plt.text(
        1,
        -0.125,
        source_text,
        ha="right",
        va="bottom",
        transform=ax5.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )

    plt.tight_layout()
    mo.mpl.interactive(ax5)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### _CHART LIBELLE AVIS "NON RECONNUE" PER COMMUNES (%)_
    """)
    return


@app.cell
def _(mo):
    dropdown_pct = mo.ui.dropdown(
        options=["25%", "50%", "75%", "100%"], value="75%", label="Select percentage"
    )
    dropdown_pct
    return (dropdown_pct,)


@app.cell
def _(details_data, dropdown_pct, mo, pd):
    libelle_avis_non_commune = details_data.copy()
    libelle_avis_non_commune["dateArrete"] = pd.to_datetime(
        libelle_avis_non_commune["dateArrete"]
    )

    libelle_avis_non_commune = libelle_avis_non_commune[
        libelle_avis_non_commune["dateArrete"].dt.year > 1983
    ]
    libelle_avis_non_commune = (
        libelle_avis_non_commune.groupby(["codeInsee", "libelleAvis"])
        .count()
        .reset_index(level=1)
    )

    # Check if this is correct
    pivot_commune = libelle_avis_non_commune.pivot_table(
        index="codeInsee",
        columns="libelleAvis",
        values="nomCommune",
        aggfunc="sum",
        fill_value=0,
    )

    total_per_communes = pivot_commune.sum(axis=1)
    non_reconnue_pct = (pivot_commune["Non reconnue"] / total_per_communes) * 100

    # Nombre de commune >75% Non reconnus
    non_reconnue_pct_select = non_reconnue_pct[
        non_reconnue_pct >= int(dropdown_pct.selected_key.split("%")[0])
    ]
    pct_value = non_reconnue_pct_select.shape[0] / non_reconnue_pct.shape[0] * 100
    mo.md(f"Pourcentage de Communes no reconnues à ce taux: {round(pct_value, 2)}%")
    return (non_reconnue_pct,)


@app.cell
def _(ax3, fig_arrow, font, mo, non_reconnue_pct, plt, sns, source_text):
    _, ax6 = plt.subplots(figsize=(10, 6))

    sns.histplot(non_reconnue_pct, color="skyblue", bins=20)

    plt.title("Distribution du taux de Non Reconnaissance", fontsize=20, pad=20)
    plt.text(
        0.5,
        1.01,
        "(Après 1983)",
        font=font,
        fontsize=11,
        ha="center",
        transform=ax6.transAxes,
    )

    plt.xlabel("Taux (%)", fontsize=12)
    plt.ylabel("Nombre de Communes", fontsize=12)

    plt.xticks(font=font, rotation=0, fontsize=9)
    plt.yticks(font=font, fontsize=10)

    fig_arrow(
        head_position=(0.85, 0.15),
        tail_position=(0.8, 0.35),
        width=2,
        radius=-0.2,
        color="darkred",
        fill_head=False,
        mutation_scale=1,
    )

    non_reconnue_pct_100 = non_reconnue_pct[non_reconnue_pct == 100]
    plt.text(
        0.9,
        0.45,
        f"Aucune arrête reconnue pour $\mathbf{{{non_reconnue_pct_100.shape[0]}}}$ communes",
        ha="right",
        va="bottom",
        transform=ax3.transAxes,
        fontsize=11,
        color="darkred",
        font=font,
    )
    plt.text(
        0.15,
        -0.075,
        source_text,
        ha="right",
        va="bottom",
        transform=ax3.transAxes,
        fontsize=8,
        color="black",
        font=font,
    )

    mo.mpl.interactive(ax6)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Analyse temporelle totale par departement et region
    """)
    return


@app.cell
def _(gpd):
    regions_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson"
    dept_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    communes_geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes.geojson"

    regions_geo = gpd.read_file(regions_geojson_url)
    dept_geo = gpd.read_file(dept_geojson_url)
    communes_geo = gpd.read_file(communes_geojson_url)
    return (communes_geo,)


@app.cell
def _():
    return


@app.cell
def _():
    # Analyse temporelle totale, par commune, departement et region
    # Final Carto avec bivariate nombre d'arrete et reconnaissance
    # https://github.com/mikhailsirenko/bivariate-choropleth/blob/main/bivariate-choropleth.ipynb
    # https://www.kaggle.com/code/yotkadata/bivariate-choropleth-map-using-plotly
    # Make an analysis like this: https://www.linkedin.com/feed/update/urn:li:activity:7422549430869958657/
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CARTO
    """)
    return


@app.cell
def _(communes_geo, font, mpatches, non_reconnue_pct, plt):
    # Increase the figure size
    _, ax_com_geo = plt.subplots(figsize=(12, 8))

    non_reconnue_pct_ = non_reconnue_pct.to_frame(name="taux_non_reconnaissance")
    communes_geo_non_rec = communes_geo.merge(
        non_reconnue_pct_, left_on="code", right_on="codeInsee", how="left"
    )
    communes_geo_non_rec.plot(
        ax=ax_com_geo,
        column="taux_non_reconnaissance",
        cmap="Reds",
        legend=True,
        legend_kwds={"label": " (%)", "shrink": 0.5},
        missing_kwds={
            "color": "grey",
            "edgecolor": "white",
            "label": "Données non disponibles",
        },
        edgecolor="black",
        linewidth=0.1,
    )

    null_patch = mpatches.Patch(color="grey", label="Inconnues / Pas de cas")
    ax_com_geo.legend(handles=[null_patch], loc="lower left", frameon=False)

    # 3. Aesthetics
    ax_com_geo.set_axis_off()
    ax_com_geo.set_facecolor("none")
    ax_com_geo.set_title(
        "Taux de non-reconnaissance par commune", font=font, fontsize=20, color="black"
    )

    plt.show()
    return (communes_geo_non_rec,)


@app.cell
def _(communes_geo_non_rec, details_data, pd):
    catnat_commune = details_data.copy()
    catnat_commune["dateArrete"] = pd.to_datetime(catnat_commune["dateArrete"])

    catnat_commune = catnat_commune[catnat_commune["dateArrete"].dt.year > 1983]
    catnat_commune = catnat_commune.groupby(["codeInsee"]).count()[["nomCommune"]]
    catnat_commune = catnat_commune.rename(columns={"nomCommune": "nombre_catnat"})

    commune_geo_bivar = communes_geo_non_rec.merge(
        catnat_commune, left_on="code", right_on="codeInsee", how="left"
    )

    # Normalize Taux de non-reconnaissance
    target_col = "taux_non_reconnaissance"
    t_min = commune_geo_bivar[target_col].min()
    t_max = commune_geo_bivar[target_col].max()

    commune_geo_bivar[target_col + "_norm"] = (
        commune_geo_bivar[target_col].fillna(t_min) - t_min
    ) / (t_max - t_min)

    # Normalize Nombre de CatNat
    target_col_2 = "nombre_catnat"
    c_min = commune_geo_bivar[target_col_2].min()
    c_max = commune_geo_bivar[target_col_2].max()

    commune_geo_bivar[target_col_2 + "_norm"] = (
        commune_geo_bivar[target_col_2].fillna(c_min) - c_min
    ) / (c_max - c_min)
    return (commune_geo_bivar,)


@app.cell
def _(commune_geo_bivar, plt, sns):
    percentiles_q1 = commune_geo_bivar["taux_non_reconnaissance"].quantile([0.7, 0.9])
    p70_q1 = percentiles_q1[0.7]
    p90_q1 = percentiles_q1[0.9]

    _, axq1 = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=commune_geo_bivar,
        x="taux_non_reconnaissance",
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        ax=axq1,
    )

    axq1.axvline(p70_q1, color="orange", linestyle="--", label=f"70% ({p70_q1:.1f}%)")
    axq1.axvline(p90_q1, color="red", linestyle="--", label=f"90% ({p90_q1:.1f}%)")

    axq1.axhline(0.7, color="gray", alpha=0.3, linestyle=":")
    axq1.axhline(0.9, color="gray", alpha=0.3, linestyle=":")

    axq1.set_title("Distribution cumulative du taux de non-reconnaissance")
    axq1.set_xlabel("Taux (%)")
    axq1.set_ylabel("Probabilité cumulative (Densité)")
    axq1.legend()

    plt.show()
    return


@app.cell
def _(commune_geo_bivar, plt, sns):
    percentiles_q2 = commune_geo_bivar["nombre_catnat"].quantile([0.5, 0.85])
    p70_q2 = percentiles_q2[0.5]
    p90_q2 = percentiles_q2[0.85]

    _, axq2 = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=commune_geo_bivar,
        x="nombre_catnat",
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        ax=axq2,
    )

    axq2.axvline(p70_q2, color="orange", linestyle="--", label=f"50% ({p70_q2:.0f})")
    axq2.axvline(p90_q2, color="red", linestyle="--", label=f"85% ({p90_q2:.0f})")

    axq2.axhline(0.7, color="gray", alpha=0.3, linestyle=":")
    axq2.axhline(0.9, color="gray", alpha=0.3, linestyle=":")

    axq2.set_title("Distribution cumulative du nombre de CatNat")
    axq2.set_xlabel("Nombre")
    axq2.set_ylabel("Probabilité cumulative (Densité)")
    axq2.legend()

    plt.show()
    return


@app.cell
def _(commune_geo_bivar, matplotlib, np, pd):
    percentiles_n1 = commune_geo_bivar["taux_non_reconnaissance_norm"].quantile(
        [0.7, 0.9]
    )
    percentiles_n2 = commune_geo_bivar["nombre_catnat_norm"].quantile([0.5, 0.85])

    p_n1 = [percentiles_n1[0.7], percentiles_n1[0.9]]
    p_n2 = [percentiles_n2[0.5], percentiles_n2[0.85]]

    bins_var1 = [0, p_n1[0], p_n1[1], 1]
    bins_var2 = [0, p_n2[0], p_n2[1], 1]

    commune_geo_bivar["Var1_Class"] = pd.cut(
        commune_geo_bivar["taux_non_reconnaissance_norm"],
        bins=bins_var1,
        include_lowest=True,
    )
    commune_geo_bivar["Var1_Class"] = commune_geo_bivar["Var1_Class"].astype("str")

    commune_geo_bivar["Var2_Class"] = pd.cut(
        commune_geo_bivar["nombre_catnat_norm"], bins=bins_var2, include_lowest=True
    )
    commune_geo_bivar["Var2_Class"] = commune_geo_bivar["Var2_Class"].astype("str")

    # Code created x bins to 1, 2, 3
    x_class_codes = np.arange(1, len(bins_var1))
    d = dict(
        zip(
            commune_geo_bivar["Var1_Class"].value_counts().sort_index().index,
            x_class_codes,
        )
    )
    commune_geo_bivar["Var1_Class"] = commune_geo_bivar["Var1_Class"].replace(d)

    # Code created y bins to A, B, C
    y_class_codes = ["A", "B", "C"]
    d = dict(
        zip(
            commune_geo_bivar["Var2_Class"].value_counts().sort_index().index,
            y_class_codes,
        )
    )
    commune_geo_bivar["Var2_Class"] = commune_geo_bivar["Var2_Class"].replace(d)

    # Combine x and y codes to create Bi_Class
    commune_geo_bivar["Bi_Class"] = (
        commune_geo_bivar["Var1_Class"].astype("str") + commune_geo_bivar["Var2_Class"]
    )

    print(
        "Number of unique elements in Var1_Class =",
        len(commune_geo_bivar["Var1_Class"].unique()),
    )
    print(
        "Number of unique elements in Var2_Class =",
        len(commune_geo_bivar["Var2_Class"].unique()),
    )
    print(
        "Number of unique elements in Bi_Class =",
        len(commune_geo_bivar["Bi_Class"].unique()),
    )

    all_colors = [
        "#e8e8e8",
        "#b0d5df",
        "#64acbe",
        "#e4acac",
        "#ad9ea5",
        "#627f8c",
        "#c85a5a",
        "#985356",
        "#574249",
    ]
    cmap = matplotlib.colors.ListedColormap(all_colors)
    cmap
    return all_colors, cmap


@app.cell
def _(all_colors, cmap, commune_geo_bivar, font, plt):
    fig_bi, ax_bi = plt.subplots(figsize=(8, 8))

    # Step 1: Draw the map
    commune_geo_bivar.plot(
        ax=ax_bi, column="Bi_Class", cmap=cmap, categorical=True, legend=False
    )

    plt.tight_layout()
    plt.axis("off")
    ax_bi.set_title(
        "Communes plus affectées par CatNat et non reconnunes",
        font=font,
        fontsize=20,
        color="black",
    )

    img2_bi = fig_bi  # refer to the main figure
    ax2_bi = fig_bi.add_axes([0.1, 0.2, 0.1, 0.1])

    alpha = 1

    # Column 1
    ax2_bi.axvspan(
        xmin=0, xmax=0.33, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[0]
    )
    ax2_bi.axvspan(
        xmin=0, xmax=0.33, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[1]
    )
    ax2_bi.axvspan(
        xmin=0, xmax=0.33, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[2]
    )

    # Column 2
    ax2_bi.axvspan(
        xmin=0.33, xmax=0.66, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[3]
    )
    ax2_bi.axvspan(
        xmin=0.33, xmax=0.66, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[4]
    )
    ax2_bi.axvspan(
        xmin=0.33, xmax=0.66, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[5]
    )

    # Column 3
    ax2_bi.axvspan(
        xmin=0.66, xmax=1, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[6]
    )
    ax2_bi.axvspan(
        xmin=0.66, xmax=1, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[7]
    )
    ax2_bi.axvspan(
        xmin=0.66, xmax=1, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[8]
    )

    # Step 3: annoate the legend
    ax2_bi.tick_params(
        axis="both", which="both", length=0
    )  # remove ticks from the big box
    ax2_bi.axis("off")  # turn off its axis
    ax2_bi.annotate(
        "", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
    )  # draw arrow for x
    ax2_bi.annotate(
        "", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
    )  # draw arrow for y
    ax2_bi.text(
        s="Taux de non-reconnaissance", x=0.1, y=-0.25, color="black"
    )  # annotate x axis
    ax2_bi.text(
        s="Nombre de CatNat", x=-0.25, y=0.1, rotation=90, color="black"
    )  # annotate y axis
    plt.show()
    return (alpha,)


@app.cell
def _(all_colors, alpha, cmap, commune_geo_bivar, font, plt):
    fig_bi_high, ax_bi_high = plt.subplots(figsize=(8, 8))

    # 1. Background: Draw all communes with low opacity (alpha)
    commune_geo_bivar.plot(
        ax=ax_bi_high,
        column="Bi_Class",
        cmap=cmap,
        categorical=True,
        alpha=0.65,
        edgecolor="none",
    )
    plt.tight_layout()
    plt.axis("off")
    ax_bi_high.set_title(
        "Focus : Zones à risque et non-reconnaissance élevées",
        font=font,
        fontsize=20,
        color="black",
    )

    # Highlight Top-right category
    darkest_label = "3C"
    high_high = commune_geo_bivar[commune_geo_bivar["Bi_Class"] == darkest_label]

    high_high.plot(
        ax=ax_bi_high,
        color=all_colors[8],  # The darkest color (top-right of legend)
        edgecolor="black",
        linewidth=1,
        alpha=1.0,
    )

    img2_bi_high = fig_bi_high  # refer to the main figure
    ax2_bi_high = fig_bi_high.add_axes([0.1, 0.2, 0.1, 0.1])

    # Column 1
    ax2_bi_high.axvspan(
        xmin=0, xmax=0.33, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[0]
    )
    ax2_bi_high.axvspan(
        xmin=0, xmax=0.33, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[1]
    )
    ax2_bi_high.axvspan(
        xmin=0, xmax=0.33, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[2]
    )

    # Column 2
    ax2_bi_high.axvspan(
        xmin=0.33, xmax=0.66, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[3]
    )
    ax2_bi_high.axvspan(
        xmin=0.33, xmax=0.66, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[4]
    )
    ax2_bi_high.axvspan(
        xmin=0.33, xmax=0.66, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[5]
    )

    # Column 3
    ax2_bi_high.axvspan(
        xmin=0.66, xmax=1, ymin=0, ymax=0.33, alpha=alpha, color=all_colors[6]
    )
    ax2_bi_high.axvspan(
        xmin=0.66, xmax=1, ymin=0.33, ymax=0.66, alpha=alpha, color=all_colors[7]
    )
    ax2_bi_high.axvspan(
        xmin=0.66, xmax=1, ymin=0.66, ymax=1, alpha=alpha, color=all_colors[8]
    )

    # Step 3: annoate the legend
    ax2_bi_high.tick_params(
        axis="both", which="both", length=0
    )  # remove ticks from the big box
    ax2_bi_high.axis("off")  # turn off its axis
    ax2_bi_high.annotate(
        "", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
    )  # draw arrow for x
    ax2_bi_high.annotate(
        "", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)
    )  # draw arrow for y
    ax2_bi_high.text(
        s="Taux de non-reconnaissance", x=0.1, y=-0.25, color="black"
    )  # annotate x axis
    ax2_bi_high.text(
        s="Nombre de CatNat", x=-0.25, y=0.1, rotation=90, color="black"
    )  # annotate y axis

    # Add a highlight indicator on the legend itself
    ax2_bi_high.add_patch(
        plt.Rectangle((0.66, 0.66), 0.34, 0.34, fill=False, edgecolor="black", lw=2)
    )

    plt.show()
    return


@app.cell
def _(details_data, pd):
    # Franchise

    franchise_commune = details_data.copy()
    franchise_commune["dateArrete"] = pd.to_datetime(franchise_commune["dateArrete"])

    franchise_commune = franchise_commune[franchise_commune["dateArrete"].dt.year > 200]
    franchise_commune = franchise_commune[
        franchise_commune["libelleAvis"] != "Non reconnue"
    ]

    # Testing query
    is_increased = franchise_commune["franchise"].isin(
        ["Doublée", "Triplée", "Quadruplée"]
    )

    franchise_commune["is_increase"] = is_increased.astype(int)

    hypothesis_df = (
        franchise_commune.groupby("codeInsee")
        .agg(total_catnat=("franchise", "count"), nb_increases=("is_increase", "sum"))
        .reset_index()
    )

    # Calculate the ratio (frequency of increase per event)
    hypothesis_df["increase_rate"] = (
        hypothesis_df["nb_increases"] / hypothesis_df["total_catnat"]
    )

    # franchise_commune = franchise_commune.groupby(["codeInsee"]).count()[["nomCommune"]]
    # franchise_commune = franchise_commune.rename(columns={"nomCommune":"nombre_catnat"})

    # commune_geo_bivar2 = communes_geo_non_rec.merge(catnat_commune, left_on="code", right_on="codeInsee", how="left")

    # franchise_commune["franchise"] = franchise_commune["franchise"].replace({"-":np.nan,"Simple":1,"Doublée":2,"Triplée":3,"Quadruplée":4})

    """
    franchise_commune = franchise_commune[franchise_commune["franchise"].isin(["Doublée","Triplée","Quadruplée"])]
    pivot_franchise = franchise_commune.pivot_table(
        index="codeInsee", 
        columns="franchise", 
        aggfunc="size", 
        fill_value=0
    )
    pivot_franchise

    df_long = pivot_franchise.melt(var_name="Niveau", value_name="Nombre")
    df_long
    """
    return (hypothesis_df,)


@app.cell
def _(hypothesis_df, plt, sns):
    _, ax_franc = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=hypothesis_df, x="total_events", y="increase_rate", ax=ax_franc
    )

    ax_franc.set_title(
        "Hypothèse : Plus de CatNat entraîne-t-il plus de franchises doublées ?"
    )
    ax_franc.set_xlabel("Nombre total d'arrêtés CatNat par commune")
    ax_franc.set_ylabel("Taux de franchise augmentée")

    plt.show()
    return


@app.cell
def _():
    # Analyse par departement et region
    # Analyse par type de cata
    # Model de classification
    return


if __name__ == "__main__":
    app.run()
