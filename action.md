# MMchange Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/MMchange`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/MMchange
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/MMchange
git switch reproduce
git pull --ff-only
```

## Environment and pretrained weights

```bash
bash setup_server_env.sh
conda activate mmchange-rs
bash prepare_server_data.sh /home/xxj/Data
```

Setup installs PyTorch 2.7.1/CUDA 12.8, OpenAI CLIP, and Transformers 4.40.1; it also caches CLIP ViT-B/32 and the ResNet-50 ImageNet checkpoint. The TinyLLaVA model is separate: download it from the [project's Quark link](https://pan.quark.cn/s/8e1a70eda144), extract it on the server, and use its directory for caption generation.

## Generate captions

MMchange needs a text description for each image in both time steps, for train, val, and test. Generate one JSON per dataset before training:

```bash
bash generate_text.sh levircd /path/to/TinyLLaVA
bash generate_text.sh levircdplus /path/to/TinyLLaVA
bash generate_text.sh cdd /path/to/TinyLLaVA
bash generate_text.sh whucd /path/to/TinyLLaVA
bash generate_text.sh sysucd /path/to/TinyLLaVA
```

The script resumes from existing entries and writes captions under `MMchange_work_dirs/<dataset>/captions.json`. This pass runs TinyLLaVA once per T1/T2 image and can take a substantial amount of time.

## Train

After generating captions for a dataset:

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers preserve the code defaults: 256 input, 40,000 max steps, batch 32, Adam, lr `5e-4`, poly schedule, and weight decay `1e-4`. Override runtime settings with `GPU_ID=1` or `NUM_WORKERS=4`.

## Test

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

The wrapper uses the corresponding `best_model.pth`; an explicit checkpoint path can be supplied as a second argument. Test also requires the same dataset captions JSON. Metrics and logs are written beneath `MMchange_work_dirs/<dataset>/`.
