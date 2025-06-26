import argparse
import cluster_algos
import data_loader
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

def get_pmts_pos(i,output_tree, pmt_dict):
    """
    Gets the position of the photo-sensors that detected light for a particular event (i).

    Parameters
    ----------
    i : Integer
        The event number in the ROOT file.
    output_tree : ROOT tree
        Contains the variables from the simulation data.
    pmt_dict : dictionary
        The map linking photo-sensor ID to the position.

    Returns
    -------
    points : numpy array
        An array containting the x and y positions


    """
    pmtIDs = ak.to_numpy(output_tree["digitPMTID"].array()[i])

    unique_points = set()
    
    for id in pmtIDs:
        x = pmt_dict[id][0]
        y = pmt_dict[id][1]
        unique_points.add((x,y))

    # Convert to numpy array for clustering
    points = np.array(list(unique_points))

    return points

def plot_sipm_pos(positions, title, labels=None, centers=None, noise=None, is_clustered=False):

    if not is_clustered:
        plt.scatter(positions[:, 0], positions[:, 1], c='black', alpha=0.6)
        
    else:
        plt.scatter(positions[:, 0], positions[:, 1], c=labels, cmap='viridis', alpha=0.6)
        if noise is not None:
            plt.scatter(positions[noise, 0], positions[noise, 1], c='red', marker='x', label='Noise')
        if len(centers)>0:
            plt.scatter(centers[:, 0], centers[:, 1], s=300, c='blue', marker='X', label='Centers')
            plt.legend()

    # Add dashed circle representing detector size
    circle = plt.Circle((0, 0), 900, color='blue', linestyle='--', fill=False)
    plt.gca().add_artist(circle)

    # Set custom x and y axis ranges
    plt.xlim(-950, 950)
    plt.ylim(-950, 950)
    
    plt.title(title)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.show()

def cluster_data(pmt_dict, output_tree, eps=30, min_samples=5, apply_energy_cut=True, apply_position_cut=False):
    nentries = output_tree.num_entries
    # Loop over simulation data, apply DBSCAN and do a simple analysis
    for i in range(0,nentries):
        if i>5: # only want to check a few events
            break

        # get the energy of the particle
        energy = (ak.to_numpy(output_tree["scintEdepQuenched"].array()[i]))[0]

        # get the position of the particle
        mcx = (ak.to_numpy(output_tree["mcx"].array()[i]))[0]
        mcy = (ak.to_numpy(output_tree["mcy"].array()[i]))[0]
        mcz = (ak.to_numpy(output_tree["mcz"].array()[i]))[0]

        rho = np.sqrt(mcx*mcx+mcy*mcy)

        # Apply energy and position cuts conditionally
        energy_cut = (1.5 < energy < 2.0) if apply_energy_cut else True
        position_cut = (-1500 < mcz < 1500 and rho < 500) if apply_position_cut else True

        if energy_cut and position_cut:
            points = get_pmts_pos(i,output_tree, pmt_dict)
            # plot_sipm_pos(points, f'SiPM positions, event {i}') # just checking the positions

            dbscan_labels, dbscan_centers = cluster_algos.apply_dbscan(points, eps=eps, min_samples=min_samples)

            # mini analysis
            npoints = len(points)
            unique_labels = np.unique(dbscan_labels)
            unique_labels = len(unique_labels) - 1 # removing the noise label
            nclustered = np.count_nonzero(dbscan_labels > -1)
            print(f'Clustered {nclustered} out of {npoints} SiPMs, in {unique_labels} clusters.')

            # plot result
            plot_sipm_pos(points, f'DBSCAN Clustering event {i}, Energy {energy:.2f} MeV', labels=dbscan_labels, centers=dbscan_centers, is_clustered=True)

def main():
    
    parser = argparse.ArgumentParser(description="Process ROOT file for SiPM clustering.")
    parser.add_argument("file_path", type=str, help="Path to the ROOT file")
    args = parser.parse_args()

    # Open and load ROOT file
    pmt_dict, output_tree = data_loader.read_root_file(args.file_path)
        
    cluster_data(pmt_dict, output_tree)
    
if __name__ == "__main__":
    main()

