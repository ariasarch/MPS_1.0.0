cd _desired_location_

git clone git@github.com:ariasarch/GUI_PSS_0.0.1.git

cd GUI_PSS_0.0.1

conda install -c conda-forge mamba

conda create -n PSS_env python=3.8.15 -y

conda activate PSS_env

mamba install -c conda-forge -y cvxpy==1.2.1 dask==2021.2.0 ffmpeg-python==0.2.0 matplotlib==3.2.2 networkx==2.4 numba==0.52.0 numpy==1.20.2 pandas==1.2.3 Pillow==8.2.0 psutil==5.9.5 pyfftw==0.12.0 pymetis==2020.1 rechunker==0.3.3 scipy==1.9.1 scikit-image==0.18.1 scikit-learn==0.22.1 SimpleITK==2.0.2 sparse==0.11.2 xarray==0.17.0 zarr==2.16.1 distributed==2021.2.0 medpy==0.4.0 natsort==8.4.0 statsmodels==0.13.2 tifffile==2020.6.3 tqdm==4.66.1

pip install opencv-python==4.2.0.34 --no-deps
