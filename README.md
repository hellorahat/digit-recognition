This program runs several pipelines to determine which combination of feature extraction method yields the best results for handwritten digit recognition.

**main.py** serves as the main entrypoint to the program; running all pipelines and saving the models.

**pipeline.py** includes several pipelines which use different combinations of feature extraction strategies:
- Pipeline 1: HOG
- Pipeline 2: HOG + Sparse Zoning
- Pipeline 3: HOG + Dense Zoning
- Pipeline 4: HOG + Sparse Zoning + Projection Histogram

**features/hog.py** implements the Histogram of Oriented Gradients algorithm from scratch.
**features/zoning.py** implements the Zoning algorithm from scratch.

**model/DecisionTree.py** implements the Decision Tree Classifier from scratch. Splits are computed based on Gini impurity.<br>
**model/RandomForestClassifier.py** implements a Random Forest Classifier from scratch. The classifier uses bootstrap samping and random feature selection.