# ============================================
# ZOO DATASET - DECISION TREE CLASSIFICATION
# ============================================

# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================
# VERI SETINI YUKLEME
# ============================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

columns = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "type"
]

df = pd.read_csv(url, names=columns)

# İlk 5 satır
print("===== İLK 5 SATIR =====")
print(df.head())

print("\n===== VERİ SETİ BOYUTU =====")
print(df.shape)

# ============================================
# HAYVAN İSİMLERİNİ SAKLAMA
# ============================================

animal_names = df["animal_name"]

# ============================================
# GEREKSIZ SUTUNU SILME
# ============================================

df = df.drop("animal_name", axis=1)

# ============================================
# FEATURE VE TARGET AYIRMA
# ============================================

X = df.drop("type", axis=1)
y = df["type"]

# ============================================
# SINIF İSİMLERİ
# ============================================

class_names_dict = {
    1: "Mammal",
    2: "Bird",
    3: "Reptile",
    4: "Fish",
    5: "Amphibian",
    6: "Bug",
    7: "Invertebrate"
}

# ============================================
# TRAIN - TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X,
    y,
    animal_names,
    test_size=0.2,
    random_state=42
)

# ============================================
# DECISION TREE MODEL
# ============================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# TAHMIN
# ============================================

y_pred = model.predict(X_test)

# ============================================
# PERFORMANS METRIKLERI
# ============================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

print("\n===== PERFORMANS SONUÇLARI =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

labels = [
    "Mammal",
    "Bird",
    "Reptile",
    "Fish",
    "Amphibian",
    "Bug",
    "Invertebrate"
]

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================
# FEATURE IMPORTANCE GRAFİĞİ
# ============================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================
# DECISION TREE GÖRSELLEŞTİRME
# ============================================

plt.figure(figsize=(35,20))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=list(class_names_dict.values()),
    filled=True,
    fontsize=5,
    proportion=True,
    rounded=True
)

plt.title("Decision Tree", fontsize=20)

plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================
# GINI VE ENTROPY KARŞILAŞTIRMASI
# ============================================

gini_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

gini_model.fit(X_train, y_train)
entropy_model.fit(X_train, y_train)

gini_pred = gini_model.predict(X_test)
entropy_pred = entropy_model.predict(X_test)

gini_acc = accuracy_score(y_test, gini_pred)
entropy_acc = accuracy_score(y_test, entropy_pred)

print("\n===== GINI vs ENTROPY =====")
print(f"Gini Accuracy    : {gini_acc:.4f}")
print(f"Entropy Accuracy : {entropy_acc:.4f}")

# ============================================
# MAX DEPTH ANALİZİ
# ============================================

depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = []

for d in depths:

    temp_model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    temp_acc = accuracy_score(y_test, temp_pred)

    scores.append(temp_acc)

plt.figure(figsize=(8,5))

plt.plot(depths, scores, marker='o')

plt.title("Max Depth vs Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    "max_depth_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================
# YANLIŞ SINIFLANDIRILAN HAYVANLAR
# ============================================

print("\n===== YANLIŞ SINIFLANDIRILAN HAYVANLAR =====")

for i in range(len(y_test)):

    actual = y_test.iloc[i]
    predicted = y_pred[i]

    if actual != predicted:

        print(f"Hayvan: {names_test.iloc[i]}")

        print(
            f"Gerçek Sınıf : {class_names_dict[actual]}"
        )

        print(
            f"Tahmin Edilen: {class_names_dict[predicted]}"
        )

        print("------------------------")

# ============================================
# BİTİŞ
# ============================================

print("\nProje başarıyla tamamlandı.")