SpatialDiffusion
======

Code and instructions for our paper:

Automating Multimodal 3D Building Layout Generation via Text and Graph Conditioning.

<img src="https://github.com/MelodyXXX/SpatialDiffusion/blob/main/images/SpatialDiffusion.png">


Volumetric Design Process
-----
<img src="https://github.com/MelodyXXX/SpatialDiffusion/blob/main/images/VolumetricDesign.png">

Data
------
- 000000–059999 Train dataset: LLM annotation
- 060000–095999 Train dataset: rule-based annotation
- 096000–097999 Test dataset: LLM annotation
- 098000–099999 Test dataset: rule-based annotation
- 100000–100255 Validate dataset


**Raw dataset** — the raw volumetric design data is from [Building-GAN](https://github.com/AutodeskAILab/Building-GAN) (Chang et al., ICCV 2021). Download via:
```bash
cd /tmp && wget -O 6types-raw_data.zip "https://d271velnk8wqmg.cloudfront.net/6types-raw_data.zip" \
  && unzip 6types-raw_data.zip && rm 6types-raw_data.zip
```

**Annotated dataset** — download:[here](https://drive.google.com/file/d/1q_HIrYGKyknXTcY7TZ8P66tTyIgasZQ-/view?usp=sharing)

Please put the subfolders in ```Dataset/6types-raw_data/``` as below. Each volumetric design has one file in each subfolder:

```
Dataset/
└── 6types-raw_data/
    ├── global_graph_data/   # global graph: FAR, program ratios, and associated rooms for each program type
    ├── local_graph_data/    # local graph: bubble diagram (type and size of each room, connectivity between rooms)
    ├── voxel_data/          # voxel graph
    ├── caption_data/        # text captions (rule-based and LLM descriptions), e.g. caption_000000.json
    └── caption_vis/         # rendered images of the buildings
```

Running pretrained models
------

For running a pre-trained model, please follow the steps below:
- The pre-trained model can be download [here](https://drive.google.com/file/d/1aPpMuMSq6s-4swi9tE5hFEInWPTWsj9T/view?usp=sharing).
- Put the pre-trained model in `/workspaces/SpatialDiffusion/runs/aic/checkpoints`
- Run ```python 02_inference.py```
- Check out the results in the `inference/{model}/{epch_current_time}/output` folder.


Training models
------
For training a model from scratch, please follow the steps below:
- run ```python 01_train_diffusion.py ```. Customized arguments can be set according to ```utils/args.py```. 
- Check out ```output``` and ```checkpoints``` folders for intermediate outputs and checkpoints, respectively.


Citation
------
```
*(coming soon)*
```

If you use the dataset, please also cite Building-GAN:
```bibtex
@inproceedings{chang2021building,
  title     = {Building-GAN: Graph-Conditioned Architectural Volumetric Design Generation},
  author    = {Chang, Kai-Hung and Cheng, Chin-Yi and Luo, Jieliang and Murata, Shingo and Nourbakhsh, Mehdi and Tsuji, Yoshito},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {11956--11965},
  year      = {2021}
}
```


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
