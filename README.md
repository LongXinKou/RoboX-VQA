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
### Real-World Dataset
(1) First of all, download raw RLDS datasets from [Open X-Embodiment](https://console.cloud.google.com/storage/browser/gresearch/robotics) using gsutil tool.

```bash
conda install gsutil
gsutil -m cp -r gs://gresearch/robotics/<dataset_name> <path_to_your_target_dir>
```

(2) Then read RLDS format datasets and extract video sequences with annotations.

```bash
# Basic Usage
python scripts/RLDS_reader.py --dataset bridge_data_v2

# Custom Output Directory
python scripts/RLDS_reader.py --dataset bridge_data_v2 --output-dir /path/to/output

# List Available Datasets
python scripts/RLDS_reader.py --list-datasets
```

The generated directory structure shows as below:

```
dataset_name/
├── video/
│   ├── 000000.mp4
│   ├── 000001.mp4
│   ├── ...
│   ├── annotation.json
│   └── meta_information.json
```

### Meta-Information

![meta_information](./images/meta_information.png)

### Automatic QA Generation

We leverages GPT-4o’s capabilities to automatically generate comprehensive question-answer pairs from text meta-information, 
the **dual-source meta-information** provides rich contextual data for QA generation.

```
Please ask a question for {Q_type} based on the video. Then you need to give the answer. 

Video Content: {input} 

# The video depicts a robot performing tasks on a tabletop.
# Only describe what you are certain about, and avoid providing descriptions that may be ambiguous or inaccurate.
# Your response must be in JSON format, with the keys question and answer, like this: {{question: your question here, answer: your answer here}}.
```

Our prompt engineering follows a structured approach with several key components: 
- **Meta-Information Integration**: Each prompt begins by providing the available meta-information as context, ensuring GPT-4o has access to the details of the demonstration.  
- **Task Type Specification**: The prompt explicitly defines the type of understanding to be probed. 
- **Question Format Guidelines**: We provide clear instructions on question formatting. 
- **Answer Constraint Specifications**: To maintain evaluation consistency, we define answer format requirements tailored to 
each question type.



## Data Format 

## Evaluation

## Citation