import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

print("🚀 Loading dataset...")
X, y = load_iris(return_X_y=True)

print("🤖 Training model...")
clf = RandomForestClassifier()
clf.fit(X, y)

print("💾 Saving model to model.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model saved successfully!")
