from scipy.sparse import load_npz
human_DIs = []
human_meta = []
mouse_DIs = []
mouse_meta = []
output_path = "../nanni/Projects/SingleCellHiC/sparse/"
bins = pd.read_csv('../nanni/Projects/SingleCellHiC/bins_all.tsv', sep='\t')
matrices_meta = pd.read_csv('~/Dataset/MAIN/human_meta.tsv', sep='\t')
for info in matrices_meta.itertuples():
    print(info.id, info.species)
    m = load_npz(os.path.join(output_path, "{}.npz".format(info.id))).todense()
    m = m + np.triu(m, k=1).T
    s_bins = bins[bins.chr.str.contains(info.species)].sort_values(['chr', 'start', "end"])
    DIs = []
    for chrom in sorted(s_bins.chr.unique()):
        from_bin =  s_bins[s_bins.chr == chrom].bin.min()
        to_bin =  s_bins[s_bins.chr == chrom].bin.max() + 1
        mchrom = m[from_bin:to_bin, from_bin:to_bin]
        for i in range(0, mchrom.shape[1] ):
            A = np.array(mchrom[i, :i]).flatten().sum()
            B = np.array(mchrom[i, i+1:]).flatten().sum()
            DI = (B - A)/np.sqrt(A**2 + B**2)
            DI = DI if not np.isnan(DI) else 0
            DIs.append(DI)
    DIs = np.array(DIs)
    if info.species == 'human':
        human_DIs.append(DIs)
        human_meta.append(info)
        print(DIs)
    else:
        mouse_DIs.append(DIs)
        mouse_meta.append(info)
human_DIs = np.vstack(human_DIs)
np.save("human_DIs.npy", human_DIs)
