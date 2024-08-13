# Plant Traits

1. Download the dataset from kaggle and unzip the file
2. Move all contents of the data folder to the current folder: `mv data/* .`
3. Run the Jupyter Notebook `CS480_Project.ipynb`



## Troubleshooting
If 
```
import matplotlib_inline.backend_inline
```
fails, reload notebook.

If numpy dtype fails, ensure numpy version is less than $2$ (see https://stackoverflow.com/questions/78634235/numpy-dtype-size-changed-may-indicate-binary-incompatibility-expected-96-from)