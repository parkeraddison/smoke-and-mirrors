# (More Than) Smoke and Mirrors: Using Virtual Data to Fine-tune Real World Wildfire Smoke Detection

Improving wildfire smoke detection models by creating virtual training data with Unreal Engine 5.

This repository contains UE source code for creating environments to capture virtual smoke imagery, Python scripts implementing smoke detection machine learning models, and notebooks evaluating model performance when trained on real data versus fine-tuning with virtual data sources.

**Contents:**
| Path | Description |
| --- | --- |
| src/ | Source code for preprocessing and loading data, training and evaluating models, and utils |
| VirtualSmoke/ | Unreal Engine 5.0.3 project for generating virtual smoke images |
| 1_Datasets.ipynb | Downloading and organizing data sources; Instructions for generating the virtual images; Preprocessing the data |
| 2_Model.ipynb | Training a model on various combinations of data sources |
| 3_Experiment.ipynb | Replicating many trials of model training on different combinations of data sources; Analysis of the results |

## Dependencies

**Notebooks:**
- Python3 and `requirements.txt`
- [HPWREN FIgLib dataset](http://hpwren.ucsd.edu/HPWREN-FIgLib/HPWREN-FIgLib-Data/)

**Virtual Smoke Generation:**
- [Unreal Engine 5](https://www.unrealengine.com/en-US/unreal-engine-5)
  - Plugin: Cesium for Unreal
  - Plugin: Volumetrics
  - Content: UE5 Starter Content

## Acknowledgements

- **HPWREN** (https://hpwren.ucsd.edu) for consistently providing some of the best wildfire-related research and resources out there for public use and public good
- **Unreal Engine 5** (https://www.unrealengine.com/en-US/unreal-engine-5) for being such a powerful piece of free software
- The **Deloitte AI Guild** (https://www2.deloitte.com/us/en/pages/careers/articles/technology-guild-program.html) and special thanks to **Dr. Suhan Ree** for providing mentorship throughout the creation and formalization of this work
