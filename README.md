
# Sc-HiC-FeatureSelection
## Usage
### convert_to_numpy
In order to use the proposed algorithem, the input data file should be in `.npy` format. python script  `convert_to_numpy.py` converts raw data file to `.npz` format.
In `convert_to_numpy.py`, `BIN_SIZE` is the desired resolution to create the Contact Matrix. Download and unpack required gz files (`raw_dir` , `schic2_mm9_dir`  ,`chrom_sizes.txt` and ...) from [GSE94489](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94489) and  [GSE84920](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84920) .


### Directionality Index (`DI_imp.py``):

To apply the Directionality index to the contact matrices, you can use `DI_imp.py` which:
`output_path` is the path which the contact matrices are saved (with `".npy"` extension)
`bins`: chromosome region size in form of [Chr:Start-End].
matrices_meta : csv file which represents the labels for each sample( in folder matrices_meta you can find for corresponded file for  `GSE94489` and `GSE84020`)

The output of `DI_imp.py` is a numpy dataset (`human_DIs.npy`) which is the directionality index matrix of the dataset.

### Feature selection and evaluation


To run the proposed feature selection method and evaluation using MLPC, python file  `ScHiC-FS.py`` should `be run, which:

`DataMain` : The location of `human_DIs.npy`` file which is obtained in the previous step.
`human_meta` : csv file which represents the labels for each sample( in folder matrices_meta you can find for corresponded file for  `GSE94489` and `GSE84020`)

To apply method on Cell Cycle dataset (`GSE94489`) the following code should be executed:
```python
  DataMain=np.load("/home/amirreza/Dataset/MAIN/Dataset/human_DIs-cc.npy")
  data=DataMain
  cell_cycle={'G1':1, 'ES':2,'MS':3, 'G2':4}
  matrices_meta = pd.read_csv( '/home/amirreza/Dataset/MAIN/Dataset/matrices_meta_cellcycle.tsv', sep="\t")
  matrices_meta.cell_cycle=[cell_cycle[item] for item in matrices_meta.cell_cycle]
  label=matrices_meta['cell_cycle']
  label=np.array(label)
  ```
To apply method on Cell line dataset (GSE84020) the following code should be executed:
```python
  DataMain=np.load("/home/amirreza/DI_indices22/human_DIs-cL_FULL.npy")
  data=DataMain
  cell_line={'HAP1':1, 'HeLa':2,'GM12878':3, 'K562':4}
  matrices_meta = pd.read_csv(os.path.join("./DI_indices", '/home/nanni/Projects/SingleCellHiC/DI_indices/matrices_meta.tsv'), sep="\t")
  matrices_meta.cell_line=[cell_line[item] for item in matrices_meta.cell_line]
  label=matrices_meta['cell_line']
  label=np.array(label)
  ```


In order `to choose the combination method you should change the combination_method in which for AND operation :`combination_method= 1` ,OR Operation :`combination_method=2`, Majority Voting Operation :`combination_method=3` and for Random selection ``combination_method=4`.
As mentioned in the article, each feature selection method required a threshold which it can define in the threshold variable in the code.

#### Output
The outputs of this script include:
acc : The accuracy on each cross-fold validation iterations
YP : y_predict (predicted labels) in each iteration
YS: y_score (predicted probability) in each iteration
YT: true labels in each iteration
