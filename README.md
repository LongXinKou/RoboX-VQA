# RoboX-VQA Dataset

**RoboX-VQA**, a large-scale, high-quality and context-rich robotics VQA dataset across diverse scenarios.

<a href='https://roboannotatex.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2311.17043'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/koulx/roboannotatorx'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/koulx/RoboX-VQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

## Overview of RoboX-VQA
![qa_distribution](./images/qa_distribution.png)

## Data Sources
### 💡 Real World Data
The foundation of RoboX-VQA(real) builds upon Open X-Embodiment, which encompasses over 1 million real robot trajectories spanning 22 distinct robot
platforms and demonstrating 527 unique skills across 160,266 task instances. To extend this foundation, we developed two additional long-horizon manipulation datasets: bridge data v2 combine (2K trajectories) and bridge data v2 combine rss
(1K trajectories)
![oxe_distribution](./images/oxe_distribution.png)

### 🎁 Sim Data
Simulated data serves as a **complementary resource** to real-world datasets by addressing key challenges such as the scarcity of long-horizon data and sparse annotations. It enables:
- Enable the generation of **Large-scale, High-quality, Consistent data**.
- **Diversity and Richness** in training datasets, especially for rare or complex task scenarios. 
- Scalability in generating and annotating data, ensuring continuous model improvements.


## Data Prepraring
![qa_generation](./images/qa_generation.png)

## Data Format 

## Evaluation

## Citation