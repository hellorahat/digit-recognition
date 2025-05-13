"""
pipelines.py

Defines all pipelines for experimenting with different feature
extraction strategies.
"""

from features import hog, zoning
from model import RandomForest

import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.ensemble import RandomForestClassifier


def load_dataset(train_path: str, test_path: str):
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # Separate the features (pixel values) and labels (digit labels)
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)

    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int)

    return (X_train, y_train, X_test, y_test)


# Pipeline 1: HOG only
def pipeline_hog(train_path, test_path):
    # Step 1: Load the dataset
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path)

    # Step 2: Perform preprocessing and extract features
    X_train = np.array([img.reshape((16, 16)) for img in X_train])
    X_test = np.array([img.reshape((16, 16)) for img in X_test])

    # Feature extraction: extract HOG features
    X_train = np.array([hog.extract_hog_features(img,
                                                 cell_size=(4, 4),
                                                 num_bins=9)
                        for img in X_train])
    X_test = np.array([hog.extract_hog_features(img,
                                                cell_size=(4, 4),
                                                num_bins=9)
                       for img in X_test])
    # Step 3: Train classifier
    # clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf = RandomForest(num_trees=10, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[HOG Only] Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_acc in enumerate(per_class_acc):
        print(f"Class {i}: Accuracy = {class_acc:.4f}")

    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Save information
    os.makedirs("results", exist_ok=True)
    with open("results/hog_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, class_acc in enumerate(per_class_acc):
            f.write(f"Class {i}: {class_acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    return clf


# Pipeline 2: HOG + Zoning
def pipeline_hog_zoning(train_path, test_path):
    # Step 1: Load the dataset
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path)

    # Step 2: Perform preprocessing and extract features
    X_train = np.array([img.reshape((16, 16)) for img in X_train])
    X_test = np.array([img.reshape((16, 16)) for img in X_test])

    # Feature extraction: extract HOG and Zone features
    X_train_feat = []
    X_test_feat = []
    for img in X_train:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(4, 4))
        X_train_feat.append(np.concatenate([hog_feat, zone_feat]))

    for img in X_test:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(4, 4))
        X_test_feat.append(np.concatenate([hog_feat, zone_feat]))

    X_train = np.array(X_train_feat)
    X_test = np.array(X_test_feat)
    # Step 3: Train classifier
    # clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf = RandomForest(num_trees=10, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[HOG + Zoning] Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_acc in enumerate(per_class_acc):
        print(f"Class {i}: Accuracy = {class_acc:.4f}")

    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Save information
    os.makedirs("results", exist_ok=True)
    with open("results/hog_zoning_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, class_acc in enumerate(per_class_acc):
            f.write(f"Class {i}: {class_acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    return clf


# Pipeline 3: HOG + Dense Zoning
def pipeline_hog_DenseZoning(train_path, test_path):
    # Step 1: Load the dataset
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path)

    # Step 2: Perform preprocessing and extract features
    X_train = np.array([img.reshape((16, 16)) for img in X_train])
    X_test = np.array([img.reshape((16, 16)) for img in X_test])

    # Feature extraction: extract HOG and Zone features
    X_train_feat = []
    X_test_feat = []
    for img in X_train:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(2, 2))
        X_train_feat.append(np.concatenate([hog_feat, zone_feat]))

    for img in X_test:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(2, 2))
        X_test_feat.append(np.concatenate([hog_feat, zone_feat]))

    X_train = np.array(X_train_feat)
    X_test = np.array(X_test_feat)
    # Step 3: Train classifier
    # clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf = RandomForest(num_trees=10, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[HOG + Dense Zoning] Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_acc in enumerate(per_class_acc):
        print(f"Class {i}: Accuracy = {class_acc:.4f}")

    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Save information
    os.makedirs("results", exist_ok=True)
    with open("results/hog_dense_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, class_acc in enumerate(per_class_acc):
            f.write(f"Class {i}: {class_acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    return clf


# Pipeline 4: HOG + Dense Zoning + Projection Histograms
def pipeline_hog_DenseZoning_projection(train_path, test_path):
    # Step 1: Load the dataset
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path)

    # Step 2: Perform preprocessing and extract features
    X_train = np.array([img.reshape((16, 16)) for img in X_train])
    X_test = np.array([img.reshape((16, 16)) for img in X_test])

    # Feature extraction: HOG and Zone and Projection Histogram features
    X_train_feat = []
    X_test_feat = []
    for img in X_train:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(2, 2))
        stats_feat = np.concatenate([np.sum(img, axis=1), np.sum(img, axis=0)])
        X_train_feat.append(np.concatenate([hog_feat, zone_feat, stats_feat]))

    for img in X_test:
        hog_feat = hog.extract_hog_features(img, cell_size=(4, 4), num_bins=9)
        zone_feat = zoning.extract_zoning_features(img, grid_size=(2, 2))
        stats_feat = np.concatenate([np.sum(img, axis=1), np.sum(img, axis=0)])
        X_test_feat.append(np.concatenate([hog_feat, zone_feat, stats_feat]))

    X_train = np.array(X_train_feat)
    X_test = np.array(X_test_feat)
    # Step 3: Train classifier
    # clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf = RandomForest(num_trees=10, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[HOG + Dense Zoning + Projection Histograms] Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, class_acc in enumerate(per_class_acc):
        print(f"Class {i}: Accuracy = {class_acc:.4f}")

    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Save information
    os.makedirs("results", exist_ok=True)
    with open("results/hog_proj_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, class_acc in enumerate(per_class_acc):
            f.write(f"Class {i}: {class_acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    return clf
