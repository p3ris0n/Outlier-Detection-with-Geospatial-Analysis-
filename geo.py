import pandas as pd
from opencage.geocoder import OpenCageGeocode
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sb
import folium

API_KEY = "17096f1a864f4fc0a89597872933225e"
geocoder = OpenCageGeocode(API_KEY)  # initializing geocoder

# Load the datasets
main_df = pd.read_csv('KOGI_crosschecked.csv')
lat_long_df = pd.read_csv('polling-units.csv')

# Initialize a common column for both datasets
common_column = 'Ward'

# Combine dataframes
combined_df = pd.merge(main_df, lat_long_df, on=common_column)
print(combined_df)

# Save the updated dataframe to a CSV file
combined_df.to_csv("KOGI_crosschecked_coordinates.csv", index=False)

# Replace infinite values with NaN and drop rows with NaN values in latitude and longitude columns
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['location.latitude', 'location.longitude'])

combined_df['location.latitude'] = pd.to_numeric(combined_df['location.latitude'], errors='coerce')
combined_df['location.longitude'] = pd.to_numeric(combined_df['location.longitude'], errors='coerce')

combined_df = combined_df.dropna(subset=['location.latitude', 'location.longitude'])

# Locate neighbouring polling units
locations = list(zip(combined_df['location.latitude'], combined_df['location.longitude']))

neighbour_radius = 1.0  # radius in kilometers

def geodesic_dist(locations, radius_km):
    n = len(locations)  # Returns the number of geographic locations
    radius_in_degrees = radius_km / 111.0  # Convert radius from kilometers to degrees (~111km per degree)
    tree = cKDTree(locations)  # cKDTree for fast spatial queries

    neighbours = {}  # Initialize dictionary to store neighbour polling units locations

    for i, location in enumerate(locations):
        indices = tree.query_ball_point(location, radius_in_degrees)
        neighbours[i] = [j for j in indices if j != i]

    return neighbours

neighbours = geodesic_dist(locations, neighbour_radius)  # Neighbours polling units within specified radius
neighbours_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in neighbours.items()]))
neighbours_df.to_csv('KOGI_polling_distances.csv', index=False)
print(neighbours_df.head())

# Calculate the outlier scores
outlier_scores = []

parties = ['APC', 'LP', 'PDP', 'NNPP']

for pu, neighbour_indices in neighbours.items():
    try:
        pu = int(pu)  # Ensure pu is treated as an integer
        pu_votes = combined_df.iloc[pu][parties].values
        neighbour_votes = combined_df.iloc[neighbour_indices][parties].values

        if len(neighbour_votes) > 0:
            mean_neighbour_votes = np.mean(neighbour_votes, axis=0)
            std_neighbour_votes = np.std(neighbour_votes, axis=0)
            if np.any(std_neighbour_votes == 0):
                std_neighbour_votes[std_neighbour_votes == 0] = 1  # Avoid zero division error

            z_scores = (pu_votes - mean_neighbour_votes) / std_neighbour_votes
            outlier_scores.append({'Ward': pu, 'outlier_scores': z_scores})
    except KeyError as e:
        print(f"KeyError for polling unit {pu}: {e}")
    except Exception as e:
        print(f"Unexpected error for polling unit {pu}: {e}")

outlier_scores_df = pd.DataFrame(outlier_scores)

# Add the outlier scores of each polling unit for the respective parties
outlier_scores_df = pd.concat([outlier_scores_df.drop(['outlier_scores'], axis=1),
                               pd.DataFrame(outlier_scores_df['outlier_scores'].tolist(), columns=parties)], axis=1)

# Save outlier score for each party to an Excel sheet.
with pd.ExcelWriter('KOGI_outlier_scores.xlsx') as writer:
    for k in parties:
        sorted_df = outlier_scores_df.sort_values(by=k, ascending=False)
        sorted_df.to_excel(writer, sheet_name=f"Top Outliers {k}", index=False)

# Sort by outlier scores for each party
top_outliers = {}
for party in parties:
    sorted_df = outlier_scores_df.sort_values(by=party, ascending=False)
    top_3_outliers = sorted_df.head(3)
    top_outliers[party] = top_3_outliers

for party, top_3_outliers in top_outliers.items():
    print(f"Top 3 Outliers for {party}:")
    print(top_3_outliers)
    print()

    for index, row in top_3_outliers.iterrows():
        pu = int(row['Ward'])  # Ensure pu is treated as an integer
        closest_pu_indices = neighbours[pu]
        closest_pus = combined_df.iloc[closest_pu_indices]
        print(f"polling unit {pu}:")
        print(f"votes: {combined_df.iloc[pu][parties].to_dict()}")
        print(f"closest polling units:\n{closest_pus[common_column].to_list()}")
        print(f"outlier score: {row[party]}")
        print("reason: Significant deviation in votes compared to neighboring units.\n")

# visualization using folium.
m = folium.Map(location=[combined_df['location.latitude'].mean(), combined_df['location.longitude'].mean()],
               zoom_start=10)

# adding the polling units to the map.
for k, row in combined_df.iterrows():
    folium.Marker(location=[row['location.latitude'], row['location.longitude']],
                  popup=f"PU: {row['Ward']}, votes: {row[parties].to_dict()}").add_to(m)

m.save('KOGI_PU_maps.html')

# visualization with matplotlib and seaborn.
for k in parties:
    plt.figure(figsize=(10, 6))
    sb.barplot(x='Ward', y=k, data=top_outliers[k])
    plt.title(f"top 3 outliers for {k}")
    plt.xlabel('polling unit')
    plt.ylabel('outlier score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Top_3_outliers_{k}.png")
    plt.show()

outlier_scores_df.to_csv("KOGI_outlier_scores.csv", index=False)
print(outlier_scores_df)
