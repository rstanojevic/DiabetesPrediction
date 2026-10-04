from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src import config
from src.data_loader import load_raw, replace_zeros_with_nan
from src.plotting import setup_plotting, save_figure


def report_overview(df):
    print("=" * 60)
    print("OSNOVNI PREGLED")
    print("=" * 60)
    print(f"Broj redova:    {df.shape[0]}")
    print(f"Broj kolona:    {df.shape[1]}")
    print(f"Kolone:         {list(df.columns)}")
    print("\nPrvih 5 redova:")
    print(df.head())
    print("\nTipovi podataka:")
    df.info()
    print("\nDeskriptivna statistika:")
    print(df.describe().T)

def report_missing(df_raw):
    print("=" * 60)
    print("NEMOGUCE NULE / NEDOSTAJUCE VREDNOSTI")
    print("=" * 60)

    df_nan = replace_zeros_with_nan(df_raw)

    for column in config.ZERO_AS_MISSING:
        count = df_nan[column].isna().sum()
        share = count / len(df_nan) * 100
        print(f"{column:<16} {count:>4}  ({share:5.2f}%)")

    complete_rows = df_nan[config.ZERO_AS_MISSING].notna().all(axis=1).sum()
    print(f"\nRedova bez i jedne nedostajuce vrednosti: {complete_rows}")
    print(f"Izgubljeno brisanjem:   {len(df_nan) - complete_rows}")

def report_target(df):
    print("=" * 60)
    print("RASPODELA CILJNE PROMENLJIVE")
    print("=" * 60)
    counts = df[config.TARGET].value_counts().sort_index()
    shares = df[config.TARGET].value_counts(normalize=True).sort_index() * 100
    for label in counts.index:
        print(f"Outcome = {label}:  {counts[label]:>4}  ({shares[label]:5.2f}%)")

def report_complete_case_bias(df_raw):
    print("=" * 60)
    print("ANALIZA POTPUNIH SLUCAJEVA / PRISTRASNOST BRISANJA NEPOTPUNIH REDOVA")
    print("=" * 60)

    df_nan = replace_zeros_with_nan(df_raw)
    complete_mask = df_nan[config.ZERO_AS_MISSING].notna().all(axis=1)
    df_complete = df_nan[complete_mask]

    share_all = df_raw[config.TARGET].value_counts(normalize=True).sort_index() * 100
    share_complete = df_complete[config.TARGET].value_counts(normalize=True).sort_index() * 100

    print(f"{'':<20}{'ceo skup':>12}{'potpuni':>12}{'razlika':>12}")
    print(f"{'Broj uzoraka':<20}{len(df_raw):>12}{len(df_complete):>12}"
          f"{len(df_complete) - len(df_raw):>12}")

    for label in share_all.index:
        diff = share_complete[label] - share_all[label]
        name = f"Outcome = {label} (%)"
        print(f"{name:<20}{share_all[label]:>12.2f}"
              f"{share_complete[label]:>12.2f}{diff:>+12.2f}")

def report_missingness_pattern(df_raw):
    print("=" * 60)
    print("OBRAZAC NEDOSTAJANJA - INSULIN")
    print("=" * 60)

    df_nan = replace_zeros_with_nan(df_raw)

    missing_mask = df_nan["Insulin"].isna()
    group_missing = df_nan[missing_mask]
    group_present = df_nan[~missing_mask]

    print(f"Grupa BEZ izmerenog insulina: {len(group_missing)} uzoraka")
    print(f"Grupa SA izmerenim insulinom: {len(group_present)} uzoraka\n")
    print(f"{'':<28}{'bez':>10}{'sa':>10}{'razlika':>12}")

    columns = [c for c in config.FEATURES if c != "Insulin"] + [config.TARGET]

    for column in columns:
        mean_missing = group_missing[column].mean()
        mean_present = group_present[column].mean()
        diff = mean_missing - mean_present
        print(f"{column:<28}{mean_missing:>10.2f}{mean_present:>10.2f}{diff:>+12.2f}")

def plot_class_distribution(df):
    counts = df[config.TARGET].value_counts().sort_index()
    shares = df[config.TARGET].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)

    ax.bar(counts.index, counts)
    ax.set_xticks(counts.index, labels = ["Nema dijabetes (0)", "Ima dijabetes (1)"])
    ax.set_title("Raspodela ciljne promenljive")
    ax.set_ylabel("Broj uzoraka")
    ax.set_ylim(0, counts.max() * 1.15)
    for label in counts.index:
        ax.text(label, counts[label] + counts.max() * 0.02, f"{counts[label]} ({shares[label]:.1f}%)" ,ha='center')

    save_figure(fig, "fig_01_class_distribution")

def plot_missing_values(df_raw):
    df_nan = replace_zeros_with_nan(df_raw)

    shares = df_nan[config.ZERO_AS_MISSING].isna().mean() * 100
    shares = shares.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)

    positions = range(len(shares))

    ax.bar(positions, shares)
    ax.set_xticks(positions)
    ax.set_xticklabels(shares.index, rotation=45, ha="right")
    ax.set_title("Raspodela nedostajucih vrednosti")
    ax.set_ylabel("Procenat nedostajucih vrednosti")
    ax.set_ylim(0, shares.max() * 1.15)

    for i, label in enumerate(shares.index):
        ax.text(i, shares[label] + shares.max() * 0.02, f"{shares[label]:.1f}%", ha='center')

    save_figure(fig, "fig_02_missing_values")


def plot_feature_histograms(df):
    df_nan = replace_zeros_with_nan(df)
    fig, axes = plt.subplots(2, 4, figsize=config.FIGSIZE_GRID)
    axes = axes.flatten()

    for i, column in enumerate(config.FEATURES):
        ax = axes[i]
        sns.histplot(
            data=df_nan,
            x=column,
            hue=config.TARGET,
            stat="density",
            common_norm=False,
            kde=True,
            element="step",
            ax=ax,
        )
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel("Gustina" if i % 4 == 0 else "")

        if i != 0:
            ax.get_legend().remove()

    legend = axes[0].get_legend()
    legend.set_title("Outcome")
    for text, new_label in zip(legend.texts, ["Nema dijabetes (0)", "Ima dijabetes (1)"]):
        text.set_text(new_label)

    fig.suptitle("Raspodela atributa po klasi", fontsize=config.FONT_SIZE + 4)
    fig.tight_layout()
    save_figure(fig, "fig_03_feature_histograms")

def plot_feature_boxplots(df):
    df_nan = replace_zeros_with_nan(df)
    fig, axes = plt.subplots(2, 4, figsize=config.FIGSIZE_GRID)
    axes = axes.flatten()

    for i, column in enumerate(config.FEATURES):
        ax = axes[i]
        sns.boxplot(
            data=df_nan,
            hue=config.TARGET,
            legend = False,
            x=config.TARGET,
            y=column,
            ax=ax,
        )
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Nema dijabetes (0)", "Ima dijabetes (1)"])

    fig.suptitle("Raspodela atributa po klasi (boxplot)", fontsize=config.FONT_SIZE + 4)
    fig.tight_layout()
    save_figure(fig, "fig_04_feature_boxplots")

def plot_correlation_matrix(df):
    df_nan = replace_zeros_with_nan(df)
    columns = config.FEATURES + [config.TARGET]
    corr = df_nan[columns].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pirsonov koeficijent korelacije"},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Korelaciona matrica atributa")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    save_figure(fig, "fig_05_correlation_matrix")
def main():
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    setup_plotting()

    df = load_raw()
    report_overview(df)
    report_missing(df)
    report_target(df)
    report_complete_case_bias(df)
    report_missingness_pattern(df)

    plot_class_distribution(df)
    plot_missing_values(df)
    plot_feature_histograms(df)
    plot_feature_boxplots(df)
    plot_correlation_matrix(df)

if __name__ == "__main__":
    main()

