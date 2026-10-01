# HHR

Official implementation of **HHR: Hierarchical Hash Retrieval for Efficient LLM Generation**.

## Abstract

Efficient long-context inference is essential for large language models (LLMs), yet it poses a severe computational bottleneck. Hash-based retrieval offers an efficient alternative by encoding queries and keys into binary codes and using Hamming distance for key selection. However, this leads to a critical mismatch between Hamming distance and attention relevance. Query-Key logits depend jointly on directional similarity and feature magnitudes, whereas hash binarization discards magnitude information, causing both false-positive retrieval of low-logit keys and false-negative omission of high-logit keys.

To address these failures, we propose **Hierarchical Hash Retrieval (HHR)**, a coarse-to-fine framework that progressively improves retrieval accuracy through **Geometry-Aware Key Routing (GKR)** and **Learned Hash Projection (LHP)**. GKR learns a head-wise orthogonal transformation to redistribute feature magnitudes and derive more discriminative page-level logit bounds, enabling effective pruning of low-logit keys while preserving important candidates. LHP then learns a head-wise projection space that aligns Hamming distance with the true Query-Key relevance ranking for fine-grained retrieval.

By combining GKR and LHP, HHR suppresses false positives and recovers false negatives, substantially improving the fidelity of hash-based sparse attention. Extensive experiments across diverse LLMs and benchmarks demonstrate that HHR achieves superior performance over existing methods. On LongBench, HHR improves the average score by **1.10 points** and, at a context length of **128K**, achieves up to a **3.30× decoding speedup** and a **2.83× end-to-end speedup** for Llama-3.1-8B-Instruct.

## Framework
<img width="3672" height="1005" alt="HHR_framework_cropped" src="https://github.com/user-attachments/assets/b7d08b53-52fa-45e7-9084-a9c54dadaff3" />


The HHR framework consists of two stages: **Geometry-Aware Key Routing (GKR)** for coarse candidate pruning and **Learned Hash Projection (LHP)** for fine-grained hash retrieval.

## Environment

We recommend Linux with Python 3.10+ and a CUDA-capable GPU.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Build the CUDA extension:

```bash
bash scripts/fetch_third_party.sh
bash scripts/build_extension.sh
```

Check the environment:

```bash
python3 scripts/check_environment.py
```

## Training

Prepare the required training tensors and specify their location through `TRAINING_INPUT_ROOT`.

```bash
TRAINING_INPUT_ROOT=/path/to/prepared-inputs \
OUTPUT_ROOT=/path/to/hhr-run \
DEVICE=0 \
PYTHON_BIN=python3 \
bash scripts/run_train_hhr.sh
```

 
## Acknowledgements

We thank the authors of **HATA** for open-sourcing their work and implementation. Parts of this codebase were developed with reference to the HATA implementation, which greatly facilitated our development and evaluation.
