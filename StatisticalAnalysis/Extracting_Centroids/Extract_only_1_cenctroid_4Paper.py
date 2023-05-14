'''The code takes different centroids for the same particle on different 2D slices
   and create 1 single centroid for the particle in 3D.
   It saves it in a final CSV file that contains number of the particle and coordinates of each centroid.
'''

import os
import pandas as pd
import sys

name_of_retina = sys.argv[1] #'retina1' #
universel_path = sys.argv[2]

print('We are Detecting only one centroid for', name_of_retina)

# Read the centroids from the original CSV file
csv_file = universel_path + name_of_retina + '/3D_images/MyCentroids.csv'

df = pd.read_csv(csv_file)

# Prepare the output CSV file
csv_out_file = universel_path + name_of_retina + '/3D_images/My_one_time_Centroids.csv'
if os.path.exists(csv_out_file):
    os.remove(csv_out_file)
    print("Existing file deleted.")
else:
    print("File does not exist.")
my_header = True

# Iterate over unique labels
my_label_list = df['label'].unique()
for label in my_label_list:
    # Filter the DataFrame for the current label
    resu = df.loc[df['label'] == label]

    # Calculate the centroid coordinates
    centroid_img_idx = resu.iloc[int(len(resu) / 2), 0]
    moyx = resu['x_coord'].mean()
    moyy = resu['y_coord'].mean()

    # Create a new DataFrame for the centroid
    new_df = pd.DataFrame({'image_idx': [str(centroid_img_idx).zfill(4) + '.png'],
                           'label': [label],
                           'y_coord': [int(moyy)],
                           'x_coord': [int(moyx)]})

    # Append the centroid information to the output CSV file
    new_df.to_csv(csv_out_file, mode='a', header=my_header, index=False)
    my_header = False



