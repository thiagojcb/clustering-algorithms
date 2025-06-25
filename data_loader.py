import argparse
import uproot
import awkward as ak

def read_root_file(file_path):
    """
    Open and load data from a ROOT file containing two trees ('meta' and 'output') and make a SiPM map (id to position).

    Parameters:
        file_path (str): Path to the ROOT file.

    Returns:
        pmtID2pos: dictionary mapping the 'pmtId' to the position 'pmtX', 'pmtY', 'pmtZ'.
        output_tree: the ROOT tree cointaining the simulation events and related variables.
    """
    file = uproot.open(file_path)
    meta_tree = file["meta"]
    output_tree = file["output"]

    pmtID = ak.to_numpy(meta_tree["pmtId"].array()[0])
    pmtX  = ak.to_numpy(meta_tree["pmtX"].array()[0])
    pmtY  = ak.to_numpy(meta_tree["pmtY"].array()[0])
    pmtZ  = ak.to_numpy(meta_tree["pmtZ"].array()[0])

    pmtID2pos = {id: (x, y, z) for id, x, y, z in zip(pmtID, pmtX, pmtY, pmtZ)}

    return pmtID2pos, output_tree

def main():
    
    parser = argparse.ArgumentParser(description="Open ROOT file and get PMT map.")
    parser.add_argument("file_path", type=str, help="Path to the ROOT file")
    args = parser.parse_args()

    pmt_dict, output_tree = read_root_file(args.file_path)
    # print(pmt_dict)

        
if __name__ == "__main__":
    main()
