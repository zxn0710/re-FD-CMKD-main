# FD-CMKD 复现步骤

## 1. 数据集下载

```bash
git clone https://gitlab.com/cs-cooper-lab/crema-d-mirror.git
```

或

```bash
git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git
```

完整数据约 **7.55 GB**。

把下载得到的 `AudioWAV`、`VideoFlash` 两个文件夹放到：

```
FD-CMKD-main/data/CREMAD/
```

---

## 2. 安装环境包

```
python        = 3.10
torch         == 2.1.0
torchvision   == 0.16.0
librosa
pandas
numpy         < 2
opencv-python-headless == 4.9.0.80
tqdm
scipy         == 1.11.4
```

安装命令参考：

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install librosa pandas "numpy<2" opencv-python-headless==4.9.0.80 tqdm scipy==1.11.4
```

---

## 3. 数据预处理

```bash
python audio_preprocessing.py
```

> 运行前需修改 `audio_preprocessing.py` 中的 `path_to_dataset`。

```bash
python video_preprocessing.py
```

---

## 4. 修改路径

在 `CremadDataset.py` 中修改路径（**改完不用运行**）。

---

## 5. 训练

单模态训练：

```bash
python main_unimodality.py
```

> 需更改 `modality` 参数后再跑一次，以更换模态。

多模态训练：

```bash
python main_fd_cmkd.py
```


---

## 参考文献

```bibtex
@inproceedings{liu2026distilling,
  title={Distilling cross-modal knowledge via feature disentanglement},
  author={Liu, Junhong and Zhang, Yuan and Huang, Tao and Xu, Wenchao and Yang, Renyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={23739--23747},
  year={2026}
}
```
