"""
main.py

Entrypoint that runs all pipelines and saves each model.
"""
import pipelines

from concurrent.futures import ProcessPoolExecutor
import os
import pickle

TRAIN_PATH = 'data/train-data.txt'
TEST_PATH = 'data/test-data.txt'


def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    os.makedirs('saved-models', exist_ok=True)
    # Run all pipelines concurrently
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(pipelines.pipeline_hog, TRAIN_PATH, TEST_PATH): 'hog_rf.pkl',
            executor.submit(pipelines.pipeline_hog_zoning, TRAIN_PATH, TEST_PATH): 'hog_zoning_rf.pkl',
            executor.submit(pipelines.pipeline_hog_DenseZoning, TRAIN_PATH, TEST_PATH): 'hog_dense_rf.pkl',
            executor.submit(pipelines.pipeline_hog_Zoning_projection, TRAIN_PATH, TEST_PATH): 'hog_proj_rf.pkl'
        }

        for future, filename in futures.items():
            try:
                # Save all models
                clf = future.result()
                save_model(clf, f'saved-models/{filename}')
                print(f"Saved model to models/{filename}")
            except Exception as e:
                print(f"[ERROR] Pipeline failed: {e}")
