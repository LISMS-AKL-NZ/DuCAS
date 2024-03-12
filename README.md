# DuCAS: a knowledge-enhanced dual-hand compositional action segmentation method for human-robot collaborative assembly

This repository is the official repository made for the paper **DuCAS: a knowledge-enhanced dual-hand compositional action segmentation method for human-robot collaborative assembly**. It contains the necessary data and code to replicate DuCAS.

## Data preparation
The data is hosted in [Dropbox](https://www.dropbox.com/scl/fo/314g59o3qlplkwk2zfvyh/h?rlkey=yow1900rkxf5pukv4a7svt9rl&dl=0). We created the dataset by selecting a sub-dataset from [HA-ViD](https://iai-hrc.github.io/ha-vid). In this repository, we only provide the features and labels that are necessary to replicate DuHa. More information about the dataset **HA-ViD** can be found at the [website](https://iai-hrc.github.io/ha-vid).
You should download the data and put it in a folder `./data`. 
The structure of `data` should look like:
```
data
├── data_av
│   ├── train_features
│   ├── train_edge_indices
│   ├── train_i3d_features
│   ├── train_lh_labels
│   ├── train_rh_labels
│   ├── test_features
│   ├── test_edge_indices
│   ├── test_i3d_features
│   ├── test_lh_labels
│   ├── test_rh_labels
├── data_mo
│   ├── ...
├── data_to
│   ├── ...
├── data_tl
│   ├── ...
```

In DuCAS, we have four action elements: action verb `av`, manipulated object `mo`, target object `to`, tool `tl`. We create a subfolder for each action element where we provide the features (bboxes for all object), edge_indices (edges between objects in the graph we introduced in paper), i3d features (scene features in the paper), lh_labels (action labels for left hand), and rh_labels (action labels for right hand) for both training set and testing set.

## Environment preparation
We provide the `environment.yml` file to help you set up the environment easily. Please change the `prefix` to your anaconda location.
* run `conda env create -f environment.yml`

## Training and testing DuCAS
To simplify the process, we use one script `main.py` to automatically train and test DuCAS. We test DuCAS after each epoch. In the HA-ViD, the videos have `front`, `side` and `top` views, are denoted as `M0` `S1` and `S2` respectively. To run the script `main.py`, please specify the action element `element`, view of the video `view` and data dictionary `data_root`.
* run `python main.py --element av --view M0 --data_root ./data/data_av/`

## Output
We save `best models`, and the `action predictions` and corresponding `confidence scores` of the `best model` in the dictionary `./output`. You can use these output files to conduct knowledge-enhanced reasoning to refine the combinations of the four action elements. 

## Check logs
The log files will be save in `./log` dictionary. It contains dual-hand action segmentation accuracy of each epoch. 

## Citation
If you find our code useful, please cite our paper. 
```
@inproceedings{
  author    = {Hao Zheng and
               Regina Lee and
               Huachang Liang and
               Yuqian Lu and 
               Xun Xu},
  title     = {DuCAS: a knowledge-enhanced dual-hand compositional action segmentation method for human-robot collaborative assembly},
  journal = {}
}
```

## Contact
If you have any question about DuCAS, please contact Hao Zheng via [email](mailto:hzhe951@aucklanduni.ac.nz).
