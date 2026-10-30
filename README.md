# OFA-CS

This repository provides the code for the paper **"One-for-All Community Search on Unseen Graphs"** (SIGIR 2026).

OFA-CS learns a community-search model on source graphs and directly applies the frozen model to unseen target graphs without target-side retraining or fine-tuning. The released example follows the original code path and runs one cross-graph experiment:

```text
Photo (source training) -> Citeseer (unseen-target evaluation)
```

## Method overview

The example includes the main OFA-CS components:

1. spectral-aware feature alignment;
2. local and multi-hop global token construction;
3. Transformer-based representation encoding;
4. diffusion-based representation refinement;
5. modularity-driven community search.

## Repository structure

```text
OFA-CS/
|-- run_example.py       # Photo-to-Citeseer training and evaluation entry
|-- aa_utils.py          # data preparation, alignment, search, and metrics
|-- model.py             # OFA-CS model
|-- layer.py             # Transformer layers
|-- lr.py                # learning-rate scheduler
|-- data_loader.py       # optional raw-data loader
|-- dataset/             # downloaded demonstration data
|-- results/             # generated checkpoints and summaries
|-- requirements.txt
`-- CITATION.cff
```

## Environment

Python 3.10 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The bundled demonstration uses preprocessed tensor caches and does not require DGL. DGL is only needed when rebuilding supported datasets from raw DGL objects through `data_loader.py`.

## Demonstration data

Download `OFA-CS_Photo_Citeseer_Data_20261030.zip` from the repository's **Releases** page and extract it into the repository root:

```bash
unzip OFA-CS_Photo_Citeseer_Data_20261030.zip -d .
```

The resulting layout must be:

```text
dataset/
|-- photo_d64_khop4_Nl10_pos10_neg10.pt
|-- citeseer_d64_khop4_Nl10_pos10_neg10.pt
`-- citeseer/
    |-- citeseer.query
    `-- citeseer.gt
```

Raw `photo_dgl.pt` and `citeseer_pyg.pt` files are not required because the example directly loads the released preprocessed caches. File hashes are listed in [dataset/README.md](dataset/README.md).

## Run the Photo-to-Citeseer example

From the repository root, run:

```bash
python run_example.py \
  --device cuda:0 \
  --epochs 10 \
  --batch_size 10000 \
  --seed 666 \
  --save_path ./results
```

The program trains on Photo and evaluates the resulting representations on Citeseer. It reports F1, NMI, ARI, and Jaccard scores in the console. Citeseer query and ground-truth files are used only for metric reporting and do not participate in model optimization.

After completion, the following files are written to `results/`:

```text
ofacs_photo_to_citeseer_seed666_epoch10.pt
photo_to_citeseer_summary.json
```

The target-side scores printed during execution are diagnostics only; they are not used to update the model or select a checkpoint.

## Citation

```bibtex
@inproceedings{li2026ofacs,
  title     = {One-for-All Community Search on Unseen Graphs},
  author    = {Li, Mo and Zhao, Zhaosong and Ding, Linlin and Borovica-Gajic, Renata and Yao, Zhongming and Li, Jianxin},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2026},
  doi       = {10.1145/3805712.3809594}
}
```

## License

A source-code license has not yet been specified. Please contact the paper authors before redistributing or reusing the code.
