import os
import math
import scipy
# import torchaudio
import numpy as np
# import pandas as pd



# directory and file names
DTW_DIR = "DTWData/"
YES_TEMP = "yes_template.txt"
NO_TEMP  = "no_template.txt" 
YES_VAL  = "yes_validation.txt"
NO_VAL   = "no_validation.txt" 

MYSTERY_1 = "mystery1.txt" 
MYSTERY_2 = "mystery2.txt"

# all txt files to loop through and compare dtw distances
ALL_FILES = [YES_TEMP, YES_VAL, NO_TEMP, NO_VAL, MYSTERY_1, MYSTERY_2]


def euclid_dist(template, sample):
    return scipy.spatial.distance.euclidean(template, sample)

def DTW_matrix(template_file, sample_file):

    # print(f"performing DTW on {template_file}, {sample_file}")

    temp_path = os.path.join(DTW_DIR, template_file)
    samp_path  = os.path.join(DTW_DIR, sample_file)

    template_vecs = np.loadtxt(temp_path)
    sample_vecs   = np.loadtxt(samp_path)
    #print(template_vecs.shape)
    #print(sample_vecs.shape)


    dtw = np.zeros((template_vecs.shape[0],sample_vecs.shape[0])) # init empty matrix
    dtw[0,0] = euclid_dist(template_vecs[0], sample_vecs[0]) # get first cell

    for i in range(1, template_vecs.shape[0]):
        # fills out the first column
        dtw[i, 0] = dtw[i-1, 0] + euclid_dist(template_vecs[i], sample_vecs[0])

    for i in range(1, sample_vecs.shape[0]):
        # fills out the first row
        dtw[0, i] = dtw[0, i] + euclid_dist(template_vecs[0], sample_vecs[i])


    # now fill in every cell in the matrix D by calculating the sum of distances leading up to that cell.
    # based off of slide 34's example:
    # D(temp[i], samp[j]) = min(D(temp[i-1], samp[j]),
    #                          D(temp[i], samp[j-1]),
    #                           D(temp[i-1], samp[j-1),
    #                           + euclid_dist(temp[i],samp[j])

    for i in range(1, template_vecs.shape[0]):
        for j in range(1, sample_vecs.shape[0]):
            dtw[i,j] = min(dtw[i-1, j], 
                           dtw[i,j-1], 
                           dtw[i-1,j-1]) + euclid_dist(template_vecs[i],sample_vecs[j])
            
    #print(dtw.shape)
    #print(dtw)
    # the last element is the distance
    print(f"performing DTW on {template_file.replace(".txt", "")}, {sample_file.replace(".txt", "")}: {dtw[-1,-1]}")
    return 


def main():
    for file1 in ALL_FILES:
        for file2 in ALL_FILES:
            DTW_matrix(file1, file2)
    return


if __name__ == "__main__":
    main()





