# PBR-Net
🔥 Image relighting with albedo, normal , etc ...   

## 🔧 Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/foxbeing7/PBR-Net.git
    cd PBR-Net
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```
---

## Datasets
    coming soon..


## 🔥test results
<p align="center">
  <img src="/samples/masked.png" width="256" height="256" style="margin-right: 10px;">
  <img src="/samples/albedo.png" width="256" height="256" style="margin-right: 10px;">
  <img src="/samples/normal.png" width="256" height="256">
</p>
<p align="center">
  <img src="/samples/huge/masked.png" width="256" height="256" style="margin-right: 10px;">
  <img src="/samples/huge/albedo.png" width="256" height="256" style="margin-right: 10px;">
  <img src="/samples/huge/normal.png" width="256" height="256">
</p>
😊rendered result
<p align="center">
  <img src="dora.gif" width="1000" height="500" style="margin-right: 10px;">
</p>  

😊Download pre-trained model from Google drive: [Lumos_att.pth](https://drive.google.com/drive/folders/1HFVA0P-Ho8sDMiP1w5ORTN7X0wI-5Xcd?usp=sharing) .then put into ./ckpts  

😊Then config your data path and run predict_mask.py to get albedo,normal...

If PBR-Net is helpful, please help to ⭐ this repo or recommend it to your friends.  

## 📧 Contact
If you have any question, feel free to email: foxbeing@hotmail.com
