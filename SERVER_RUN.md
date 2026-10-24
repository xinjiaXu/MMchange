# MMchange server setup

MMchange requires three inputs per image pair: the two images, binary mask, and a text description for each timestamp. Images and masks are read directly from `/home/xxj/Data`; no data replicas or list files are created. The caption JSON is stored with this model's work outputs.

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/MMchange`:

```bash
bash setup_server_env.sh
conda activate mmchange-rs
```

The repository README specifies Python 3.8 and PyTorch 1.8. This server environment uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for RTX 5090 compatibility. `requirements_server.txt` includes OpenAI CLIP, Transformers 4.40.1, Tokenizers 0.19.0, and the model's other runtime packages. Setup prefetches the CLIP ViT-B/32 weights and the ResNet-50 ImageNet weights referenced by the model. The latter is changed from the repository's author-machine path to the same-named official PyTorch checkpoint URL.

The environment setup requires internet access for the OpenAI CLIP package and its ViT-B/32 weights, and for the official ResNet-50 checkpoint. TinyLLaVA weights are separate and are not downloaded by setup.

## Data layout and captions

Check the five data roots:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

The loader uses existing `train.txt`, `val.txt`, and `test.txt` files when found beside their split folders. Otherwise it enumerates each split's label folder. The text JSON uses absolute image paths as keys so the same captions can be looked up by the data loader.

| Key | Dataset root | Image A | Image B | Mask |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

Download and extract the TinyLLaVA model linked from the project README. Generate one captions file per dataset:

```bash
bash generate_text.sh levircd /path/to/TinyLLaVA
bash generate_text.sh levircdplus /path/to/TinyLLaVA
bash generate_text.sh cdd /path/to/TinyLLaVA
bash generate_text.sh whucd /path/to/TinyLLaVA
bash generate_text.sh sysucd /path/to/TinyLLaVA
```

Each command processes T1 and T2 images in train, val, and test, then saves/resumes captions at `MMchange_work_dirs/<key>/captions.json`. This text-generation pass is required before training and can take substantial time because it runs TinyLLaVA on every image.

## Train and test

The training defaults in code are 256x256 inputs, 40,000 max steps, batch size 32, Adam with lr `5e-4`, polynomial learning-rate decay, and weight decay `1e-4`.

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

Override GPU or runtime resources with `GPU_ID=1` or `NUM_WORKERS=4`. Checkpoints and logs go under `MMchange_work_dirs/<key>/<key>_iter_40000_lr_0.0005/`. The training flow evaluates validation during training and then tests its best-F1 checkpoint at completion.

To evaluate a saved best model independently:

```bash
bash run_test.sh levircd
```

Or pass a checkpoint explicitly:

```bash
bash run_test.sh levircd MMchange_work_dirs/levircd/levircd_iter_40000_lr_0.0005/best_model.pth
```

The test loader accepts either a raw `best_model.pth` or a training checkpoint containing a `state_dict` entry.
