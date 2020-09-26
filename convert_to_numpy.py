import numpy as np
import os
import logging
import pandas as pd
import pybedtools as pybed
from scipy.sparse import csr_matrix, save_npz


BIN_SIZE = 500000
coordinate_names = ['chr', 'start', 'end']


root_dir = os.path.abspath(os.path.dirname(__file__))
raw_dir = os.path.join(root_dir, 'raw')
schic2_mm9_dir = os.path.join(root_dir, 'schic2_mm9')
chrom_sizes = os.path.join(schic2_mm9_dir, "trackdb", "chrom_sizes.txt")
contact_maps_path = os.path.join(root_dir, 'maps', "resolution_{}".format(BIN_SIZE))


def __order_bins(y):
    x = y.copy()
    b1, b2 = x['bin1'],x['bin2']
    x['bin1'] = min(b1, b2)
    x['bin2'] = max(b1, b2)
    return x


def adj_to_matrix(adj, fends_coords_with_bin, shape):
    adj_with_bins = adj.merge(fends_coords_with_bin[['fend', 'bin_id']], left_on='fend1', right_on='fend')
    adj_with_bins = adj_with_bins.rename(columns={'bin_id': 'bin1'})[['bin1', 'fend2', 'count']]
    adj_with_bins = adj_with_bins.merge(fends_coords_with_bin[['fend', 'bin_id']], left_on='fend2', right_on='fend')
    adj_with_bins = adj_with_bins.rename(columns={'bin_id': 'bin2'})[['bin1', 'bin2', 'count']]
    adj_with_bins = adj_with_bins.apply(__order_bins, axis=1).groupby(['bin1', 'bin2'])['count'].sum().reset_index()
    m = csr_matrix((adj_with_bins['count'].values, (adj_with_bins['bin1'].values, adj_with_bins['bin2'].values)), 
                    shape=shape)
    return m


def main():
    
    logging.info("Creating output path {}".format(contact_maps_path))
    os.makedirs(contact_maps_path, exist_ok=True)
    
    
    if os.path.isfile(os.path.join(contact_maps_path, 'bins.tsv')):
        logging.info("Loading bins")
        bins = pd.read_table(os.path.join(contact_maps_path, 'bins.tsv'))
    else:
        logging.info("Creating bins")
        bins = pybed.BedTool().window_maker(g=chrom_sizes, w=BIN_SIZE).to_dataframe(names=coordinate_names)
        bins['chr'] = 'chr' + bins.chr.astype(str)
        bins = bins.sort_values(coordinate_names).reset_index(drop=True)
        bins['bin_id'] = np.arange(bins.shape[0], dtype=int)
        bins.to_csv(os.path.join(contact_maps_path, 'bins.tsv'), sep="\t", index=False, header=True)
    
    
    if os.path.isfile(os.path.join(contact_maps_path, 'fends_coords_with_bin.tsv')):
        logging.info("Loading fends-bins mapping")
        fends_coords_with_bin = pd.read_table(os.path.join(contact_maps_path, 'fends_coords_with_bin.tsv'))
    else:
        logging.info("Loading fends coordinates")
        fends_coords = pd.read_table(os.path.join(schic2_mm9_dir, "seq", "redb", "GATC.fends"), 
                                 converters={'chr': lambda x: "chr" + str(x)})
        fends_coords = fends_coords.assign(start=lambda x: x.coord, end=lambda x: x.coord + 1).drop('coord', axis=1)
        fends_coords = fends_coords[fends_coords.columns[1:].append(fends_coords.columns[0:1])]
        fends_coords = fends_coords.sort_values(coordinate_names).reset_index(drop=True)


        logging.info("Mapping fends to bins")
        fends_coords_with_bin = pybed.BedTool.from_dataframe(fends_coords).map(pybed.BedTool.from_dataframe(bins), c=4, o='min')\
                                             .to_dataframe(names=fends_coords.columns.tolist() + ['bin_id'])
        fends_coords_with_bin.to_csv(os.path.join(contact_maps_path, 'fends_coords_with_bin.tsv'), sep="\t", index=False, header=True)
    
    
    for exp in os.listdir(raw_dir):
        output_path = os.path.join(contact_maps_path, exp)
        os.makedirs(output_path, exist_ok=True)
        logging.info("Converting {}".format(exp))
        exp_path = os.path.join(raw_dir, exp)
        exp_files = os.listdir(exp_path)
        for ef in exp_files:
            logging.info("{} - {}".format(exp, ef))
            adj = pd.read_table(os.path.join(exp_path, ef, "adj"))
            m = adj_to_matrix(adj, fends_coords_with_bin, (bins.shape[0], bins.shape[0]))
            save_npz(os.path.join(output_path, ef + ".npz"), m)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
