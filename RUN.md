## Clustering

### Installation
Installation is quite straightforward. Simply clone the repository to get started. Requirements are quite straightforward, the only one that might be a little tricky is `faiss`. Refer to this [link](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for in-depth instructions. I used conda to install `faiss-gpu` which I believe would be the easiest way to go.

### Running clustering on your dataset
This is largely a four step process.

**Step 1**: Create a folder named `custom` containing your images and place it in `datasets`. 

**Step 2**: Modify `configs/env.yml` to include the path to where you want the results. 

**Step 3**: Specify the number of clusters by filling in the `num_classes` parameter in the following files: a) `configs/pretext/moco_custom.yml`, b) `configs/scan/scan_custom.yml`, c) `configs/pretext/scan_custom.yml`. 

**Step 4**: Run `sh clusterit.sh`

### Results

Results will be placed in the directory you specified in **Step 2** above. Enjoy!
