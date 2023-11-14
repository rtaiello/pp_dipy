<div align="center">    
 
# Privacy Preserving Dipy
![Inria](https://img.shields.io/badge/-INRIA-red) 
![Eurecom](https://img.shields.io/badge/-EURECOM-blue) <br> 
<br>
</div>

## Description

This repository contains the official code of the research paper Privacy Preserving Image Registration **currently under submission** with the extnesion of the library [Dipy](https://github.com/dipy/dipy).

## How to run
### Dependecies
You'll need a working Python environment to run the code. 
The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/products/distribution)
which provides the `conda` package manager. 
Anaconda can be installed in your user directory and does not interfere with the system Python installation.
### Configuration
- Download the repository: `git clone https://github.com/rtaiello/pp_dipy.git`
- Create the environment: `conda create -n pp_dipy python=3.7`
- Activate the environment: `conda activate pp_dipy`
- Install the dependencies: `pip install -e .`

### Launch an Affine Registration with Mutual Information  - Additive MPC protocol
- Download [ADNI](https://ida.loni.usc.edu/login.jsp) dataset
- Move the desired moving and template image in the folder `pp_experiments/data/Adni`
- Modify accordingly the path of the [template](https://github.com/rtaiello/pp_dipy/blob/b80b0f16f31d2b77f6fec387ce9af357b205e9f3/pp_experiments/affine_registration.py#L63) and [moving](https://github.com/rtaiello/pp_dipy/blob/main/pp_experiments/affine_registration.py#L68)
#### Run 🚀
- `PYTHONPATH=. python3 pp_experiments/affine_registration.py`
### Launch Symmetric Diffeomorphic Registration with Normalized Cross Correlation -  Additive MPC protocol
- Download [Abdomen MR-CT](https://learn2reg.grand-challenge.org/Datasets/) dataset
- Move the first 8 patients images and labels (both modalities) in  `pp_experiments/data/AbdomenMRCT/imagesTr` and `pp_experiments/data/AbdomenMRCT/labelsTr`
#### Run 🚀
- `PYTHONPATH=. python3 pp_experiments/syn_cc_registration_3d.py`
## Authors
* **Riccardo Taiello**  - [github](https://github.com/rtaiello) - [website](https://rtaiello.github.io)
* **Melek Önen**  - [website](https://www.eurecom.fr/en/people/onen-melek)
* **Olivier Humbert**  - [LinkedIn](https://www.linkedin.com/in/olivier-humbert-b14553173/)
* **Francesco Capano**  - [github](https://github.com/fra-cap) - [LinkedIn](https://www.linkedin.com/in/francesco-capano/)
* **Marco Lorenzi**  - [website](https://marcolorenzi.github.io/)
## Contributors:
* **Riccardo Taiello**  - [github](https://github.com/rtaiello) - [website](https://rtaiello.github.io)
## Results
### Symmetric Diffeomorphic Registration
Qualitative results for symmetric diffeomorphic registration with CC over 3D medical images. The images are presented in a 3 x 4 grid, with the first row representing the axial axis, the second row the coronal axis, and the third row the sagittal axis. In the first column of each row, the moving image obtained using PET modality is shown, while in the second column, the fixed image obtained using MRI modality is displayed. The third column shows the moving image transformed using Clear, while the fourth column shows the moving image transformed using PPIR(MPC). The transformed images are highlighted by red and green frames, respectively. 
![Image Results](https://github.com/rtaiello/pp_dipy/blob/main/github_images/syn_cc_3d.png)
### Affine Registration
Qualitative results for affine registration with MI over 3D medical images. The images are presented in a 3 x 4 grid, with the first row representing the axial axis, the second row the coronal axis, and the third row the sagittal axis. In the first column of each row, the moving image obtained using PET modality is shown, while in the second column, the fixed image obtained using MRI modality is displayed. The third column shows the moving image transformed using Clear, while the fourth column shows the moving image transformed using PPIR(MPC). The transformed images are highlighted by red and green frames, respectively.
![Image Results](https://github.com/rtaiello/pp_dipy/blob/main/github_images/linear_mi_3d.png)
