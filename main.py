
import os

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from modelset import dataset as ds
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tabulate import tabulate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.spatial.distance import euclidean
pd.set_option('display.max_columns', None)
from sklearn.metrics import classification_report
# Function to encode categorical columns
def encode_categorical(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column].astype(str))
    return df

# Function to normalize numerical columns between -10 and 10
def normalize_numerical(df):
    scaler = MinMaxScaler(feature_range=(-10, 10))
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Function to handle missing values
def handle_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
        df.select_dtypes(include=[np.number])
    )
    return df

# Function to preprocess dataset
def preprocess_dataset(df, impute_strategy='mean'):
    df = encode_categorical(df)
    df = normalize_numerical(df)
    df = handle_missing_values(df, strategy=impute_strategy)
    return df

MODELSET_HOME = r"path/to/Model_set/"

# Load datasets
ecore_dataset = ds.load(MODELSET_HOME, modeltype='ecore', selected_analysis=['stats'])
#genmymodel_dataset = ds.load(MODELSET_HOME, modeltype='uml', selected_analysis=['stats'])

# Convert to DataFrames
ecore_df = preprocess_dataset(ecore_dataset.to_normalized_df())
#genmymodel_df = preprocess_dataset(genmymodel_dataset.to_normalized_df())

# Concatenate both datasets (if needed)
#full_df = pd.concat([ecore_df, genmymodel_df], ignore_index=True)
full_df = pd.DataFrame(ecore_df)
full_df = full_df.sample(n=100)
# Create pairs of models

def create_pairs_with_labels(df):
    pairs = []
    labels = []
    finding_threshold = []
    display("length of df : "+str(len(df)))

    # finding appropiate threshold value
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            #pairs.append((df.iloc[i].values, df.iloc[j].values))
            # Calculate similarity based on Euclidean distance
            dist = euclidean(df.iloc[i].values, df.iloc[j].values)
            finding_threshold.append(dist)

    threshold = sum(finding_threshold) / len(finding_threshold)
    print(" the mean threshold is : ", threshold)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pairs.append((df.iloc[i].values, df.iloc[j].values))
            # Calculate similarity based on Euclidean distance
            dist = euclidean(df.iloc[i].values, df.iloc[j].values)
            labels.append(1 if dist < threshold else 0)
    return np.array(pairs), np.array(labels)

pairs, labels = create_pairs_with_labels(full_df)
# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(pairs, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the Siamese model
def create_siamese_model(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(input)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    return Model(inputs=input, outputs=x)

# Initialize the Siamese network
input_shape = (pairs.shape[2],)  # Number of features
base_network = create_siamese_model(input_shape)

# Define inputs for the Siamese network
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

# Generate encodings
encoded_a = base_network(input_a)
encoded_b = base_network(input_b)

# Merge encodings and add final layers
merged_vector = layers.subtract([encoded_a, encoded_b])
merged_vector = layers.Dense(16, activation='relu')(merged_vector)
output = layers.Dense(1, activation='sigmoid')(merged_vector)

# Compile the model
siamese_model = Model(inputs=[input_a, input_b], outputs=output)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = siamese_model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
    epochs=30, batch_size=16
)

# Evaluate the model
y_pred_train = siamese_model.predict([X_train[:, 0], X_train[:, 1]])
y_pred_train = (y_pred_train > 0.5).astype(int)

y_pred_test = siamese_model.predict([X_test[:, 0], X_test[:, 1]])
y_pred_test = (y_pred_test > 0.5).astype(int)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred_test))
# Print available keys in history
print("Available keys in history:", history.history.keys())
print("Available keys in history:", history.history['loss'])
# Check if the loss and validation loss exist
if 'loss' in history.history and 'val_loss' in history.history:
    # Plot the graph
    plt.plot(history.history['loss'], label='Train Loss')
    plt.yscale('log')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
else:
    print("Loss data not found in history object. Check training process.")
# similarity score distribution
def plot_similarity_scores(model, pairs, num_pairs=10):
    """
    Plot similarity scores for a given number of pairs.

    Args:
    - model: Trained Siamese model.
    - pairs: Array of pairs of input data.
    - num_pairs: Number of pairs to visualize.
    """
    similarity_scores = []
    pair_indices = range(num_pairs)

    for idx in pair_indices:
        model_a = pairs[idx, 0].reshape(1, -1)
        model_b = pairs[idx, 1].reshape(1, -1)
        similarity_score = model.predict([model_a, model_b])[0][0]
        similarity_scores.append(similarity_score)

    # Plotting the similarity scores
    plt.figure(figsize=(10, 6))
    plt.bar(pair_indices, similarity_scores, color='skyblue')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores for Selected Pairs')
    plt.xticks(pair_indices)
    plt.ylim(0, 1)  # Assuming similarity scores are between 0 and 1
    plt.show()

# Example usage
plot_similarity_scores(siamese_model, X_val, num_pairs=10)

# Function to predict similarity for multiple pairs

def display_similarity_for_first_n_pairs(model, df, pairs, n=10):
    average_similarity= 0
    print(f"Displaying similarity scores for the first {n} validation pairs:\n")
    for idx in range(n):
        model_a = pairs[idx, 0].reshape(1, -1)
        model_b = pairs[idx, 1].reshape(1, -1)
        similarity_score = model.predict([model_a, model_b])[0][0]
        print(f"Pair {idx + 1}: Similarity Score = {similarity_score:.4f}")
        average_similarity +=average_similarity

    print(f"Average Similarity Score: {average_similarity/n:.4f}")
# Example usage with first 10 validation pairs

display_similarity_for_first_n_pairs(siamese_model, full_df, X_val, n=len(X_val))