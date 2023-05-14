import cc3d
import os
import pandas as pd
import sys
import numpy as np
import nrrd

def perform_connected_components(nrrd_path):
    """
    This function performs connected components analysis on a 3D volume.
    The function takes a path to the input volume as input.
    The function first reads the input volume and converts it to a NumPy array.
    The function then performs connected components analysis on the NumPy array using the `cc3d.connected_components` function.
    The function also creates a CSV file that contains the number of dimensions for each connected component.
    The function takes the following parameters:
    * `nrrd_path`: The path to the input volume.
    """
    # Create the output path.
    nrrd_name_out = 'labeled_volume.nrrd'
    # Check if the input path exists
    if not os.path.exists(nrrd_path):
        os.makedirs(nrrd_path)

    # Read the input volume.
    input_nrrd_name = 'volume.nrrd'
    print(f"Reading the input volume from '{os.path.join(nrrd_path, input_nrrd_name)}'...")
    readdata, header = nrrd.read(os.path.join(nrrd_path, input_nrrd_name))
    # Perform connected components analysis.
    print(f"Performing connected components analysis...")
    readdata = np.transpose(readdata)
    readdata = readdata.astype(int)
    labels_out = cc3d.connected_components(readdata, connectivity=26)

    N = np.max(labels_out)

    extracted_image = labels_out * (labels_out == 1)
    # Create a CSV file to store the number of dimensions for each connected component.
    my_header = True
    csv_file = os.path.join(nrrd_path, 'Label_nbre.csv')
    my_out_img = np.copy(labels_out)
    for segid in range(1, N+1):
        dim_index = 0
        for third_dim in range(512):
            my_2d_image = labels_out[third_dim, :, :]
            my_non_null = my_2d_image[~np.all(my_2d_image != segid, axis=1)]
            if my_non_null.shape[0] != 0:
                dim_index += 1

        print('we are on {}/{}'.format(segid, N))
        data = {'label': [segid], 'dimension': [dim_index]}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, mode='a', header=my_header, index=False)
        my_header = False
        if dim_index < 2:
            my_out_img[my_out_img == segid] = 0

    nrrd.write(os.path.join(nrrd_path, nrrd_name_out), my_out_img)


if __name__ == '__main__':
    name_of_retina     = sys.argv[1] #'TX12_D6_C2R'
    universel_path     = sys.argv[2]
    nrrd_out_path      = universel_path + name_of_retina + '/3D_images/'
    perform_connected_components(nrrd_out_path)