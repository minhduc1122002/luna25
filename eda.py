# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# PALETTE = {
#     0: "#4C72B0",  # Benign - blue
#     1: "#DD8452"   # Malignant - orange
# }

# sns.set_theme(style="whitegrid", font_scale=1.1)

# df = pd.read_csv("luna25.csv")

# # plt.figure(figsize=(5, 4))
# # sns.countplot(
# #     x="label",
# #     data=df,
# #     palette=PALETTE
# # )

# # plt.title("Distribution of Lesion Labels")
# # plt.xlabel("Label (0 = Benign, 1 = Malignant)")
# # plt.ylabel("Count")

# # plt.tight_layout()
# # plt.savefig("label_distribution.png", dpi=300, bbox_inches="tight")
# # plt.show()
# # plt.close()

# # lesions_per_patient = df.groupby("PatientID").size()

# # plt.figure(figsize=(6, 4))
# # sns.histplot(
# #     lesions_per_patient,
# #     bins=10,
# #     kde=True,
# #     color="#55A868"
# # )

# # plt.title("Number of Lesions per Patient")
# # plt.xlabel("Lesions per Patient")
# # plt.ylabel("Count")

# # plt.tight_layout()
# # plt.savefig("lesions_per_patient.png", dpi=300, bbox_inches="tight")
# # plt.show()
# # plt.close()

# plt.figure(figsize=(6, 4))
# sns.countplot(
#     x="Gender",
#     hue="label",
#     data=df,
#     palette=PALETTE
# )

# plt.title("Gender Distribution by Lesion Label")
# plt.xlabel("Gender")
# plt.ylabel("Count")
# plt.legend(title="Label", labels=["Benign", "Malignant"])

# plt.tight_layout()
# plt.savefig("gender_by_label.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

import re
import pandas as pd
from collections import defaultdict

log_path = "log_2026-01-06_15-57-36.txt"

# Regex patterns
fold_pattern = re.compile(r"Fold (\d+)")
train_pattern = re.compile(r"epoch (\d+) average train loss: ([0-9.]+)")
valid_pattern = re.compile(r"epoch (\d+) average valid loss: ([0-9.]+)")

# Storage
fold_data = defaultdict(lambda: {"epoch": [], "train": [], "valid": []})

current_fold = None
temp_train = {}

with open(log_path, "r") as f:
    for line in f:
        # Detect fold
        fold_match = fold_pattern.search(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            temp_train = {}
            continue

        if current_fold is None:
            continue

        # Train loss
        train_match = train_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            temp_train[epoch] = loss

        # Valid loss
        valid_match = valid_pattern.search(line)
        if valid_match:
            epoch = int(valid_match.group(1))
            loss = float(valid_match.group(2))

            # Skip fake zeros
            if loss <= 0:
                continue

            if epoch in temp_train:
                fold_data[current_fold]["epoch"].append(epoch)
                fold_data[current_fold]["train"].append(temp_train[epoch])
                fold_data[current_fold]["valid"].append(loss)


dfs = []
for fold, data in fold_data.items():
    df = pd.DataFrame(data)
    df["fold"] = fold
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

for fold in sorted(df_all["fold"].unique()):
    df_f = df_all[df_all["fold"] == fold]
    plt.plot(df_f["epoch"], df_f["train"], label=f"Fold {fold}")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss per Fold")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("valid_loss_per_fold.png", dpi=300)
plt.show()