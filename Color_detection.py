import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract dominant colors from an image
def extract_colors(image_path, k=5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: The file {image_path} was not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image)
    
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Function to train the Random Forest model
def train_color_classifier():
    try:
        df = pd.read_csv('cleaned_colors.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Error: The dataset 'cleaned_colors.csv' was not found. Ensure it is in the same folder as this script.")
    
    X = df[['R', 'G', 'B']]
    y = df['Color_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    return model

# Function to visualize extracted colors
def plot_colors(colors):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(colors)), [1]*len(colors), color=np.array(colors)/255)
    plt.xlabel("Colors")
    plt.ylabel("Frequency")
    plt.title("Dominant Colors in Image")
    plt.show()

# Main script execution
if __name__ == "__main__":
    image_path = "Color_detection\apple_fruit_powder3.jpg"  # Change to the correct image path
    
    try:
        model = train_color_classifier()
        colors = extract_colors(image_path)
        plot_colors(colors)
    except Exception as e:
        print(e)
