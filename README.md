
<div align="center">

# RATION: Entropy-Driven Task-Adaptive Visual Attention Allocation Framework for Multimodal Reasoning



</div>

## Project Overview
Multimodal Large Language Models (MLLMs) integrate visual encoders with Large Language Models (LLMs) and enable multimodal reasoning. However, for tasks that heavily rely on visual information, the model’s utilization of visual information remains unstable, which leads to reasoning failures. Prior works mainly strengthen multimodal reasoning by improving representation alignment or increasing computation. However, these methods do not explicitly characterize the differences in visual demands across tasks, making it difficult for the model to decide where and how strongly to attend to visual information. Consequently, visual attention allocation becomes a key factor that affects multimodal reasoning. To address these, we propose RATION, an entropy-driven task-adaptive visual attention allocation framework. First, we use a task routing strategy to infer the task type of each sample and identify the key layers. We use visual attention entropy as a control signal to dynamically allocate attention according to task demands. Experiments show that RATION achieves consistent performance gains across diverse reasoning tasks, datasets, and models, providing a clear direction toward more reliable multimodal reasoning.


## Install 🛠️
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval

uv pip install -e ".[all]"

pip install -r require.txt
```
## Download
All MLLMs and datesets download from [**🤗Huggingface**]

## Inference
```bash

bash ./lmms-eval/scripts/run_ration.sh
```

