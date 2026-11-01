"""
========================================================================================
PROJECT TITLE: Customer Purchase Behavior Analysis
INSTITUTION:   ITM University Gwalior
DEPARTMENT:    Computer Science / Data Science
DESCRIPTION:   End-to-end Python pipeline for loading, cleaning, analyzing, and
               visualizing customer transaction patterns using Pandas, NumPy,
               Matplotlib, and Seaborn.
========================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling for academic/professional presentation
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILE = "customer_purchase_data.csv"


# ======================================================================================
# HELPER: Generate sample data if CSV does not exist (Ensures zero-error out-of-box run)
# ======================================================================================
def create_sample_dataset_if_missing(file_path: str):
    if not os.path.exists(file_path):
        print(f"[INFO] '{file_path}' not found. Generating a representative dataset with edge cases...")
        np.random.seed(42)
        n = 1000

        categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty & Health"]
        payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]

        data = {
            "Category": np.random.choice(categories, size=n, p=[0.25, 0.30, 0.20, 0.10, 0.15]),
            "PurchaseAmount_INR": np.random.exponential(scale=2800, size=n) + 150,
            "Purchase Frequency": np.random.randint(1, 25, size=n),
            "Age": np.random.randint(18, 70, size=n),
            "PaymentMethod": np.random.choice(payment_methods, size=n, p=[0.35, 0.20, 0.25, 0.10, 0.10]),
            "DiscountPercent": np.random.choice([0, 5, 10, 15, 20, 25, 30, 50], size=n, p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.08, 0.05, 0.02])
        }

        df_synth = pd.DataFrame(data)

        # Inject some real-world messy data (whitespace, duplicates, nulls, negative values)
        df_synth.loc[10:14, "Category"] = " Electronics  "
        df_synth.loc[20:23, "PaymentMethod"] = " UPI "
        df_synth.loc[50:52, "PurchaseAmount_INR"] = -500.0  # invalid negative
        df_synth.loc[60:62, "PurchaseAmount_INR"] = 0.0     # invalid zero
        df_synth.loc[70:71, "Age"] = np.nan                 # missing age

        # Add duplicate rows
        duplicates = df_synth.iloc[100:108].copy()
        df_synth = pd.concat([df_synth, duplicates], ignore_index=True)

        df_synth.to_csv(file_path, index=False)
        print(f"[SUCCESS] Sample dataset created with {len(df_synth)} records at '{file_path}'.\n")


# ======================================================================================
# STEP 1: DATA LOADING & INITIAL INSPECTION
# ======================================================================================
def step_1_data_loading(file_path: str) -> pd.DataFrame:
    print("=" * 80)
    print("STEP 1: DATA LOADING & INITIAL INSPECTION")
    print("=" * 80)

    df = pd.read_csv(file_path)

    print("\n--- First 5 Records ---")
    print(df.head())

    print("\n--- Dataset Dimensions (Shape) ---")
    print(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")

    print("\n--- DataFrame Information & Data Types ---")
    df.info()

    print("\n--- Missing Values Count per Column ---")
    missing_counts = df.isna().sum()
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values detected.")

    print("\n--- Duplicate Rows Count ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    return df


# ======================================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ======================================================================================
def step_2_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 80)

    initial_rows = len(df)

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print(f"Sanitized Column Names: {list(df.columns)}")

    # 2. Strip whitespace from string/categorical columns
    categorical_cols = ["Category", "PaymentMethod"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3. Remove duplicate rows
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {duplicates_before} duplicate rows.")

    # 4. Numeric conversion with coercion for safety
    numeric_cols = ["PurchaseAmount_INR", "Purchase Frequency", "Age", "DiscountPercent"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop rows with missing essential fields (e.g., Category, PurchaseAmount_INR)
    df = df.dropna(subset=["Category", "PurchaseAmount_INR", "Purchase Frequency", "PaymentMethod"])

    # 6. Filter out invalid business transactions (PurchaseAmount_INR <= 0)
    invalid_amounts = (df["PurchaseAmount_INR"] <= 0).sum()
    df = df[df["PurchaseAmount_INR"] > 0].reset_index(drop=True)
    print(f"Removed {invalid_amounts} rows with zero or negative purchase amounts.")

    rows_retained = len(df)
    print(f"\nCleaning Complete: Retained {rows_retained}/{initial_rows} rows ({(rows_retained / initial_rows) * 100:.2f}%).")
    return df


# ======================================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & SUMMARY STATISTICS
# ======================================================================================
def step_3_exploratory_data_analysis(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS & SUMMARY STATISTICS")
    print("=" * 80)

    # A. Descriptive statistics for PurchaseAmount_INR
    purchase_amt = df["PurchaseAmount_INR"]
    desc_stats = pd.Series({
        "Count": purchase_amt.count(),
        "Mean": purchase_amt.mean(),
        "Median": purchase_amt.median(),
        "Std Dev": purchase_amt.std(),
        "Min": purchase_amt.min(),
        "Max": purchase_amt.max()
    }, name="PurchaseAmount_INR_Stats")

    print("\n--- Descriptive Statistics for 'PurchaseAmount_INR' (INR) ---")
    print(desc_stats.to_frame())

    # B. Group-level summary table by Category
    category_summary = df.groupby("Category").agg(
        Transaction_Count=("PurchaseAmount_INR", "count"),
        Total_Purchase_Value=("PurchaseAmount_INR", "sum"),
        Mean_Purchase_Value=("PurchaseAmount_INR", "mean"),
        Median_Purchase_Value=("PurchaseAmount_INR", "median")
    ).sort_values(by="Total_Purchase_Value", ascending=False)

    print("\n--- Category-Level Aggregated Summary ---")
    print(category_summary.round(2))

    # C. Correlation matrix for numeric features
    numeric_features = ["PurchaseAmount_INR", "Purchase Frequency", "Age", "DiscountPercent"]
    existing_numeric = [col for col in numeric_features if col in df.columns]
    corr_matrix = df[existing_numeric].corr()

    print("\n--- Correlation Matrix (Numeric Features) ---")
    print(corr_matrix.round(4))

    return desc_stats, category_summary, corr_matrix


# ======================================================================================
# STEP 4: DATA VISUALIZATION
# ======================================================================================
def step_4_data_visualization(df: pd.DataFrame, category_summary: pd.DataFrame, corr_matrix: pd.DataFrame):
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING AND SAVING DATA VISUALIZATIONS")
    print("=" * 80)

    # Palette setup
    accent_color = "#2b5c8f"
    palette = sns.color_palette("mako", n_colors=len(category_summary))

    # ----------------------------------------------------------------------------------
    # Figure 1: Distribution of Purchase Amounts (Histogram + KDE)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df["PurchaseAmount_INR"],
        kde=True,
        color="#1f77b4",
        bins=30,
        edgecolor="black",
        alpha=0.65
    )
    plt.axvline(df["PurchaseAmount_INR"].mean(), color="red", linestyle="--", linewidth=1.8, label=f"Mean: ₹{df['PurchaseAmount_INR'].mean():,.2f}")
    plt.axvline(df["PurchaseAmount_INR"].median(), color="green", linestyle="-.", linewidth=1.8, label=f"Median: ₹{df['PurchaseAmount_INR'].median():,.2f}")
    plt.title("Figure 1: Distribution of Customer Purchase Amounts (INR)", pad=15)
    plt.xlabel("Purchase Amount (INR)")
    plt.ylabel("Frequency / Transaction Count")
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "figure1_purchase_amount_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig1_path}")

    # ----------------------------------------------------------------------------------
    # Figure 2: Total Purchase Value Across Product Categories (Bar Chart)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    cat_sorted_total = category_summary.reset_index().sort_values(by="Total_Purchase_Value", ascending=False)
    ax = sns.barplot(
        data=cat_sorted_total,
        x="Category",
        y="Total_Purchase_Value",
        hue="Category",
        legend=False,
        palette="crest"
    )
    plt.title("Figure 2: Total Purchase Value Across Product Categories", pad=15)
    plt.xlabel("Product Category")
    plt.ylabel("Total Sales Volume (INR)")
    plt.xticks(rotation=20, ha="right")

    # Add value annotations on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"₹{height:,.0f}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "figure2_total_purchase_by_category.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig2_path}")

    # ----------------------------------------------------------------------------------
    # Figure 3: Purchase Frequency vs. Transaction Value (Scatter Plot with Regression Trend)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="Purchase Frequency",
        y="PurchaseAmount_INR",
        scatter_kws={"alpha": 0.45, "color": "#0d5c75", "s": 45},
        line_kws={"color": "#d62728", "linewidth": 2, "label": "Trend Line"}
    )
    plt.title("Figure 3: Purchase Frequency vs. Transaction Value (INR)", pad=15)
    plt.xlabel("Purchase Frequency (Annual / Lifetime Purchases)")
    plt.ylabel("Purchase Amount (INR)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "figure3_frequency_vs_purchase_amount.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig3_path}")

    # ----------------------------------------------------------------------------------
    # Figure 4: Transaction Counts by Payment Method (Bar Chart)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    payment_counts = df["PaymentMethod"].value_counts().reset_index()
    payment_counts.columns = ["PaymentMethod", "TransactionCount"]
    ax = sns.barplot(
        data=payment_counts,
        x="PaymentMethod",
        y="TransactionCount",
        hue="PaymentMethod",
        legend=False,
        palette="Blues_r"
    )
    plt.title("Figure 4: Transaction Counts by Payment Method", pad=15)
    plt.xlabel("Payment Method")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=15, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "figure4_transactions_by_payment_method.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig4_path}")

    # ----------------------------------------------------------------------------------
    # Figure 5: Average Transaction Value by Category (Horizontal Bar Chart)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    cat_sorted_mean = category_summary.reset_index().sort_values(by="Mean_Purchase_Value", ascending=True)
    ax = sns.barplot(
        data=cat_sorted_mean,
        y="Category",
        x="Mean_Purchase_Value",
        hue="Category",
        legend=False,
        palette="viridis"
    )
    plt.title("Figure 5: Average Transaction Value by Category (INR)", pad=15)
    plt.xlabel("Mean Purchase Value (INR)")
    plt.ylabel("Product Category")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"₹{width:,.2f}",
                    xy=(width, p.get_y() + p.get_height() / 2),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, "figure5_average_transaction_by_category.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig5_path}")

    print("\nAll 5 figures generated and successfully saved to the 'visualizations/' directory.")


# ======================================================================================
# MAIN EXECUTION ROUTINE
# ======================================================================================
def main():
    print("""
    ############################################################################
    # ITM UNIVERSITY GWALIOR - MINI PROJECT                                    #
    # CUSTOMER PURCHASE BEHAVIOR ANALYSIS                                      #
    ############################################################################
    """)

    # 0. Check or create dataset
    create_sample_dataset_if_missing(CSV_FILE)

    # 1. Loading & Inspection
    df_raw = step_1_data_loading(CSV_FILE)

    # 2. Cleaning & Preprocessing
    df_clean = step_2_data_cleaning(df_raw)

    # 3. Exploratory Data Analysis & Statistics
    desc_stats, cat_summary, corr_matrix = step_3_exploratory_data_analysis(df_clean)

    # 4. Data Visualization (5 Figures)
    step_4_data_visualization(df_clean, cat_summary, corr_matrix)

    print("\n" + "#" * 80)
    print("# PROJECT PIPELINE COMPLETED SUCCESSFULLY! READY FOR JUPYTER / REPORT SUBMISSION.")
    print("#" * 80)


if __name__ == "__main__":
    main()
