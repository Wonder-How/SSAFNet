import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import numpy as np  # For data normalization

# Load the data from a CSV file
data = pd.read_csv(r'./entropy.csv')

# Separate features and target labels
X = data.iloc[:, 1:16]
y = data.iloc[:, 0]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Multi-Layer Perceptron classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(256, 256, 128, 32), max_iter=800, activation='relu', solver='adam', random_state=42)
mlp_model.fit(X_train, y_train)

# Predict on the test data
y_pred = mlp_model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate permutation feature importance
results = permutation_importance(mlp_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Extract the mean importance and their standard deviations
importance = results.importances_mean
features = data.columns[1:16]

# Create a DataFrame to store features and their importance
feature_importance_df = pd.DataFrame({
    'Features': features,
    'Importance': importance
})

# Assuming features are ordered by leads and frequency bands
feature_importance_df['Lead'] = np.repeat(['Lead1', 'Lead2', 'Lead3'], 5)
feature_importance_df['Band'] = np.tile(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], 3)

# Set up color mapping
norm = matplotlib.colors.Normalize(vmin=feature_importance_df['Importance'].min(), vmax=feature_importance_df['Importance'].max())
colors = matplotlib.cm.viridis(norm(feature_importance_df['Importance'].values))

# Initialize a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define 3x5 grid positions
x_pos = np.repeat(range(3), 5)  # x positions corresponding to leads
y_pos = np.tile(range(5), 3)  # y positions corresponding to frequency bands
z_pos = np.zeros_like(x_pos)
dx = dy = 0.7  # Bar width and depth
dz = feature_importance_df['Importance'].values  # Bar height, which is the importance

# Draw each bar
for i in range(len(feature_importance_df)):
    ax.bar3d(x_pos[i], y_pos[i], z_pos[i], dx, dy, dz[i], color=colors[i])

ax.set_xlabel('Leads')
ax.set_ylabel('Frequency Bands')
ax.set_zlabel('Importance')
ax.set_xticks(range(3))
ax.set_yticks(range(5))
ax.set_xticklabels(['Fp1', 'Fpz', 'Fp2'])
ax.set_yticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])

# Adjust the viewing angle for better visualization
ax.view_init(elev=20)  # 'elev' controls the elevation, 'azim' controls the azimuth

# Save the figure as an SVG file
plt.savefig(r'./entropy_Importance.svg', format='svg')

plt.show()
