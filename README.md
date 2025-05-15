# Digit Recognition Experiment with Multiple Feature Extraction Strategies

This program runs several pipelines to determine which combination of feature extraction method yields the best results for handwritten digit recognition.

## File Overview

### `main.py`
Serves as the main entry point to the program, running all pipelines and saving the models.

### `pipeline.py`
Includes several pipelines which use different combinations of feature extraction strategies:

- **Pipeline 1**: HOG  
- **Pipeline 2**: HOG + Sparse Zoning  
- **Pipeline 3**: HOG + Dense Zoning  
- **Pipeline 4**: HOG + Sparse Zoning + Projection Histogram

### `features/`
- **`hog.py`**: Implements the Histogram of Oriented Gradients algorithm from scratch.  
- **`zoning.py`**: Implements the Zoning algorithm from scratch.

### `model/`
- **`DecisionTree.py`**: Implements the Decision Tree Classifier from scratch. Splits are computed based on Gini impurity.  
- **`RandomForestClassifier.py`**: Implements a Random Forest Classifier from scratch. The classifier uses bootstrap sampling and random feature selection.
