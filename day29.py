# Import Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
df = pd.read_csv('ratings.csv')
df.drop('timestamp', axis=1, inplace=True)

print("Dataset Preview:")
print(df.head())

# Step 2: Create User-Item Matrix
user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Step 3: Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Apply SVD
svd = TruncatedSVD(n_components=50, random_state=42)
matrix = user_item_matrix.values
svd.fit(matrix)

# Latent factors
user_factors = svd.transform(matrix)
item_factors = svd.components_.T
pred_matrix = np.dot(user_factors, item_factors.T)

# Step 5: Evaluate on Test Data
def get_pred_rating(user, movie):
    try:
        user_idx = user_item_matrix.index.get_loc(user)
        movie_idx = user_item_matrix.columns.get_loc(movie)
        return pred_matrix[user_idx, movie_idx]
    except KeyError:
        return np.nan

test['pred'] = test.apply(lambda row: get_pred_rating(row['userId'], row['movieId']), axis=1)
mse = mean_squared_error(test['rating'], test['pred'])
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Step 6: Make a Prediction for a Specific User and Movie
user_id = 196
movie_id = 242
predicted_rating = get_pred_rating(user_id, movie_id)
print(f"Predicted Rating for User {user_id} on Movie {movie_id}: {predicted_rating:.2f}")

# Step 7: Visualize Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=5, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
