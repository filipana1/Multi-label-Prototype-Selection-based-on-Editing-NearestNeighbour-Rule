import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import jaccard_score
# Provided dataset

# Load the dataset into a Pandas DataFrame
#df = pd.read_csv(StringIO(data), header=None)
# Load your dataset (replace 'your_dataset.csv' with the actual filename)
df_1 = pd.read_csv('d_CALL500_norm_tr1.csv', header=None)
lenth_train_set_1= len(df_1)

# Extract features (X) and labels (y)
X = df_1.iloc[:, :-174].values  # Features 
y = df_1.iloc[:, -174:].values.astype(int)  # Labels-numbers of labels

def jaccard_distance(p1, p2):
    p1_arr = list(map(float, p1))
    p2_arr = list(map(float, p2))
    try:
        dist = 1 - jaccard_score(p1_arr, p2_arr, average='binary', zero_division=1)
    except ZeroDivisionError:
        dist = 1  # Maximum dissimilarity when no common elements
    #dist = 1 - jaccard_score(p1_arr, p2_arr, average='binary')
    return dist

def edited_nearest_neighbors(X, y, k=3, output_file=None):
    # Fit k-nearest neighbors model
    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(X)

    # Find indices of neighbors for each sample
    indices = knn.kneighbors(X, return_distance=False)[:, 1:]#The return_distance=False ensures that only indices are returned, not the actual distances.

    # Identify instances to keep
    keep_indices = np.ones(len(X), dtype=bool)
    exclude_indices = []  # List to store excluded instance indices
    for i, neighbors in enumerate(indices):
        jaccard_distances = [jaccard_distance(y[i], y[n]) for n in neighbors]  # Exclude the current instance
        #print('Jaccard distances with neighbors:', jaccard_distances)
        print(f'Jaccard distances with neighbors of instance {i}: {jaccard_distances}')

        # Check if two or more Jaccard distances are equal to 1
        if sum(distance > 0.75 for distance in jaccard_distances) == 3:# under two instances jaccard distance 1 or >=2,also change value for 0.5 or 0.75
            keep_indices[i] = False
            exclude_indices.append(i)



    # Apply the ENN rule
    X_cleaned = X[keep_indices]
    y_cleaned = y[keep_indices]

    # Save cleaned dataset to CSV file without headers
    if output_file:
        feature_columns = [f"feature_{i}" for i in range(X_cleaned.shape[1])]
        label_columns = [f"label_{i}" for i in range(y_cleaned.shape[1])]
        cleaned_df = pd.DataFrame(np.column_stack((X_cleaned, y_cleaned)), columns=feature_columns + label_columns)
        cleaned_df.to_csv(output_file, index=False, header=False, mode='w')
    print('Excluded instances:', exclude_indices)
    return X_cleaned, y_cleaned

# Apply the ENN rule and save the cleaned dataset to a new CSV file
output_path = r'enn_CALL500_tr1.csv'  # Change this to the desired output path
X_cleaned_1, y_cleaned_1 = edited_nearest_neighbors(X, y, output_file=output_path)

# Calculate reduction rate
length_cleaned_df_1  = len(X_cleaned_1)
print ('length of cleaned dataset is:',length_cleaned_df_1)

print('lenth of train_set is:', lenth_train_set_1)
reduction_rate_1 = (1 - (length_cleaned_df_1  / lenth_train_set_1) )* 100
print('final Reduction Rate is:', reduction_rate_1)

#---------------------------------data-2---------------------------------------------------------
df_2 = pd.read_csv('d_CALL500_norm_tr2.csv', header=None)
lenth_train_set_2= len(df_2)

# Extract features (X) and labels (y)
X = df_2.iloc[:, :-174].values  # Features 
y = df_2.iloc[:, -174:].values.astype(int)  # Labels-numbers of labels

# Apply the ENN rule and save the cleaned dataset to a new CSV file
output_path = r'enn_CALL500_tr2.csv'  # Change this to the desired output path
X_cleaned_2, y_cleaned_2 = edited_nearest_neighbors(X, y, output_file=output_path)

# Calculate reduction rate
length_cleaned_df_2  = len(X_cleaned_2)
print ('length of cleaned dataset is:',length_cleaned_df_2)

print('lenth of train_set is:', lenth_train_set_2)
reduction_rate_2 = (1 - (length_cleaned_df_2  / lenth_train_set_2) )* 100
print('final Reduction Rate is:', reduction_rate_2)


#---------------------------------data-3---------------------------------------------------------
df_3 = pd.read_csv('d_CALL500_norm_tr3.csv', header=None)
lenth_train_set_3= len(df_3)

# Extract features (X) and labels (y)
X = df_3.iloc[:, :-174].values  # Features 
y = df_3.iloc[:, -174:].values.astype(int)  # Labels-numbers of labels

# Apply the ENN rule and save the cleaned dataset to a new CSV file
output_path = r'enn_CALL500_tr3.csv'  # Change this to the desired output path
X_cleaned_3, y_cleaned_3 = edited_nearest_neighbors(X, y, output_file=output_path)

# Calculate reduction rate
length_cleaned_df_3  = len(X_cleaned_3)
print ('length of cleaned dataset is:',length_cleaned_df_3)

print('lenth of train_set is:', lenth_train_set_3)
reduction_rate_3 = (1 - (length_cleaned_df_3  / lenth_train_set_3) )* 100
print('final Reduction Rate is:', reduction_rate_3)

#---------------------------------data-4---------------------------------------------------------
df_4 = pd.read_csv('d_CALL500_norm_tr4.csv', header=None)
lenth_train_set_4= len(df_4)

# Extract features (X) and labels (y)
X = df_4.iloc[:, :-174].values  # Features 
y = df_4.iloc[:, -174:].values.astype(int)  # Labels-numbers of labels

# Apply the ENN rule and save the cleaned dataset to a new CSV file
output_path = r'enn_CALL500_tr4.csv'  # Change this to the desired output path
X_cleaned_4, y_cleaned_4 = edited_nearest_neighbors(X, y, output_file=output_path)

# Calculate reduction rate
length_cleaned_df_4  = len(X_cleaned_4)
print ('length of cleaned dataset is:',length_cleaned_df_4)

print('lenth of train_set is:', lenth_train_set_4)
reduction_rate_4 = (1 - (length_cleaned_df_4  / lenth_train_set_4) )* 100
print('final Reduction Rate is:', reduction_rate_4)

#---------------------------------data-5---------------------------------------------------------
df_5 = pd.read_csv('d_CALL500_norm_tr5.csv', header=None)
lenth_train_set_5= len(df_5)

# Extract features (X) and labels (y)
X = df_5.iloc[:, :-174].values  # Features allagi
y = df_5.iloc[:, -174:].values.astype(int)  # Labels-numbers of labels

# Apply the ENN rule and save the cleaned dataset to a new CSV file
output_path = r'enn_CALL500_tr5.csv'  # Change this to the desired output path
X_cleaned_5, y_cleaned_5 = edited_nearest_neighbors(X, y, output_file=output_path)

# Calculate reduction rate
length_cleaned_df_5  = len(X_cleaned_5)
print ('length of cleaned dataset is:',length_cleaned_df_5)

print('lenth of train_set is:', lenth_train_set_5)
reduction_rate_5 = (1 - (length_cleaned_df_5  / lenth_train_set_5) )* 100
print('final Reduction Rate is:', reduction_rate_5)

##################################results final Reduction Rate-------------------
sum_reduction=reduction_rate_1 + reduction_rate_2 +reduction_rate_3 +reduction_rate_4 +reduction_rate_5
average_reduction_rate = sum_reduction/5.0
print('Average Reduction Rate is:', average_reduction_rate)
