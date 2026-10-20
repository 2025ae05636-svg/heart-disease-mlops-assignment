from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import clean_target, load_data

OUT = Path("artifacts/plots")
OUT.mkdir(parents=True, exist_ok=True)

df = clean_target(load_data())

# Missing-value summary
missing = df.isna().sum().sort_values(ascending=False)
missing.to_csv("artifacts/missing_values.csv")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Class Balance")
plt.xlabel("Target (0=absence, 1=presence)")
plt.tight_layout()
plt.savefig(OUT / "class_balance.png", dpi=160)
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close()

df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig(OUT / "feature_distributions.png", dpi=160)
plt.close()

print(df.describe(include="all").T)
print("\nMissing values:\n", missing)
