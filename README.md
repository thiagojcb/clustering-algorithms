# clustering-algorithms

Python script to test clustering algorithms for LiquidO data (MeV particles)

- `data_loader.py`
    : Functions to read and load the Rat-Pac simulation output.
- `cluster_algos.py`
    : Contains three clustering algorithms: k-means, DBSCAN and Gaussian Mixture Model.
- `cluster_ana.py`
    : Example function of how to use the modules above to load a ROOT file, cluster an event and do a simple analysis.

## Usage

The three scripts above should be in the same directory. On such directory, from a terminal, run:


`python cluster_ana.py /path/to/data/CLOUD_electrons_fill_FP_parallel_16mm_976.ntuple.root`

