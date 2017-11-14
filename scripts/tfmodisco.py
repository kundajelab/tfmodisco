from __future__ import division, print_function, absolute_import
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--onehot_hdf5", required=True,
                        help="Path to .hdf5 with one-hot encoded seq data."
                             +" The data should be stored under a dataset"
                             +" named 'onehot'")
    parser.add_argument("--hypothetical_contribs_hdf5", required=True,
                        help="Path to the .hdf5 with the hypothetical"
                             " contribs. The dataset names should correspond"
                             " to the different tasks to analyze")
    parser.add_argument("--clustering_config", required=True,
                        help="Path to file with clustering config") 
