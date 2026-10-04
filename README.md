
# Toolbox for the analysis of 3D OCT images of murine retina

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the code and data for the project described in the paper "A Machine Learning Framework for the Quantification of Experimental Uveitis in Murine OCT", published in *Biomedical Optics Express* ([DOI: 10.1364/BOE.489271](https://doi.org/10.1364/BOE.489271)).

## Appendix

This work was done in the context of a Master 2 final internship, conducted under the supervision of:

    - Xavier Descombre from Université Côte d’Azur, INRIA, CNRS, I3S, Sophia Antipolis, France.
    - Alin Achim from University of Bristol, Bristol, United Kingdom.

We would like to express our gratitude for their guidance, support, and valuable insights throughout the project.

## Installation

Requires Python 3 (the code was developed with Python 3.7+ and PyTorch 1.12).

```bash
git clone https://github.com/Mellak/Toolbox-for-the-analysis-of-3D-OCT-images-of-murine-retina.git
cd Toolbox-for-the-analysis-of-3D-OCT-images-of-murine-retina
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version first if you want GPU support (see https://pytorch.org).

## Usage

The repository has two parts. Several scripts contain hard-coded paths (for example `D:/DL_Project/...` or `/home/youness/...`) that you must edit to point to your own data and weights before running them.

### 1. Statistical analysis of 3D OCT volumes (`StatisticalAnalysis/`)

Every script takes two positional arguments: the retina folder name and the root directory containing all retina folders (with a trailing slash):

```bash
python <script>.py <name_of_retina> <universal_path>/
```

Each retina folder is expected to hold the 2D B-scans in `<name_of_retina>/png/`. The pipeline, in order, is:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `ParticleDetection/LCFCN_prediction.py` | Detects particles in each B-scan with an LCFCN (FCN8-VGG16) model; writes `Particules_dir/`. Needs `ParticleDetection/Weights/model_best.pth`. |
| 2 | `Extracting_Centroids/Make3DImages.py` | Stacks the 2D particle masks into `3D_images/volume.nrrd`. |
| 3 | `RetinaDetection/Detect_Retina_Classical_Approach.py` | Segments the retina layers (classical image processing); writes `masques_png/` and `Clean_D14/`. A U-Net alternative is in `RetinaDetection/DeepLearning/predict.py`. |
| 4 | `Extracting_Centroids/3D_CC.py` | 3D connected-component labelling; writes `3D_images/labeled_volume.nrrd`. |
| 5 | `Extracting_Centroids/detecting_centroids_4paper.py` | Extracts particle centroids to `3D_images/MyCentroids.csv`. |
| 6 | `Extracting_Centroids/Extract_only_1_cenctroid_4Paper.py` | Keeps one centroid per particle: `3D_images/My_one_time_Centroids.csv`. |
| 7 | `3D-KRipley/particles_retina_distance_4Paper.py` | Computes particle-to-retina distances: `3D_images/New_Distance_File.csv`. |
| 8 | `3D-KRipley/3D_K_function_of_all_Retina_on_same_figure_4Paper.py` | Plots the 3D Ripley K function of all retinas on one figure (paths configured inside the script). |

`StatisticalAnalysis/run_software.py` runs steps 1-7 for every retina folder in a directory. Edit `working_dir` and the script paths at the top of the file first.

### 2. 2D classification of OCT B-scans (`2DClassification/`)

EfficientNet-B7 classifiers. Train/test splits (5 folds) for days D2, D6 and D14 are provided in `2DClassification/Csv_files/`. Scripts are configured by editing the variables at the top (`day`, data directories, `universel_path`) and run without arguments:

```bash
python Train_Binary_Classification.py   # train the binary classifier
python Test_Binary_Classification.py    # evaluate it; writes per-image results as CSV
python measure_metrics_2classes.py      # compute metrics (confusion matrix, ROC/AUC, ...) from those CSVs
python GradCam_Binary_Classification.py # Grad-CAM visualisations
python Train_MultiClass.py              # 4-class classifier
python Test_Multicalss.py               # evaluate the 4-class classifier
```

## Citation

If you use this code, please cite:

```bibtex
@article{mellak2023machine,
  title     = {A machine learning framework for the quantification of experimental uveitis in murine {OCT}},
  author    = {Mellak, Youness and Ward, Amy and Nicholson, Lindsay and Descombes, Xavier},
  journal   = {Biomedical Optics Express},
  volume    = {14},
  number    = {7},
  pages     = {3413},
  year      = {2023},
  doi       = {10.1364/BOE.489271}
}
```

## Authors

- [@Mellak](https://github.com/Mellak)

## License

This project is licensed under the [MIT License](LICENSE).
