# p2ilf

Code repository for the **Preoperative to Intraoperative Laparoscopy Fusion (P2ILF)** @ MICCAI2022 challenge.

## Installation instructions

```angular2html
python3 -m venv p2ilf
source p2ilf/bin/activate
pip install -U pip
pip install -r requirements.txt
```

In addition, please install pytorch3D (0.2.5) with CUDA support

```angular2html
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5'
```

and install detectron2 for mask detection

```angular2html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


