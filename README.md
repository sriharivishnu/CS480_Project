# Plant Traits

1. Download the dataset from kaggle and unzip the file
2. Move all contents of the data folder to the current folder: `mv data/* .`
3. Run the Jupyter Notebook `CS480_Project.ipynb`


Note: To generate embeddings, it can take several hours depending on GPU. To train with all 5 folds of cross validation, it can take up to 15 hours on CPU. A notebook output is in this repository as reference (`CS480_Project_notebook_output.pdf`).

If present, the `train_embeddings.parquet` and `test_embeddings.parquet` contain the cached embeddings. 

## Tuning
Tuning was done with the code snippets in `tune.py`. The obtained values are now hard-coded.


## Troubleshooting
If 
```
import matplotlib_inline.backend_inline
```
fails, reload notebook.

If numpy dtype fails, ensure numpy version is less than 2 (see https://stackoverflow.com/questions/78634235/numpy-dtype-size-changed-may-indicate-binary-incompatibility-expected-96-from)