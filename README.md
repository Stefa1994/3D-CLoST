# 3D-CLoST (3D Convolution LSTM on Spatio-Temporal)

## General Information

3D-CLoST is a space-time flow prediction framework that exploits the synergy between 3D convolution and long-short-time memory networks (LSTM) to jointly learn the characteristics of space-time correlation from low to high levels.

## Installation

Requirements:
* Python >= 3.6
* Keras version 2.4.3
* Tersorflow version 2.3.0

## Usage
  - Download data from [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhCtwaiDRy5oDVIug) or [BaiduYun](http://pan.baidu.com/s/1mhIPrRE)
  - Put the h5 file of New York in "raw data" folder (*3D-CLoST/data/raw/NY*)
  - Put the four h5 files of Beijing in "raw data" folder (*3D-CLoST/data/raw/BJ*)
  - Open the data extraction file (*3D-CLoST/Notebook*) with jupyter notebook to build the dataset
  - Open terminal in the same folder (*stdn/*)
  - Run with "python main.py" for NYC taxi dataset, or "python main.py --dataset=bike" for NYC bike dataset

## License

3D_CloST is released under the MIT License (refer to the LICENSE file for details).
