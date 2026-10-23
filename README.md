# kafun

Crowdsourced pollen-source mapping. Users photograph trees, a classifier identifies Japanese cedar (*sugi*) and Japanese cypress (*hinoki*), and geotagged detections are turned into a pollen-risk heatmap and a low-pollen walking route.

Built as part of the EPBL5 course at Ritsumeikan University, using the Osaka Ibaraki Campus (OIC) area as the test site.

## Pipeline

```
photo ─▶ plant gate ─▶ close-up gate ─▶ background removal ─▶ species classifier ─▶ detections.json
         (CLIP)        (CLIP)           (BiRefNet)            (ResNet50)                 │
                                                                                         ├─▶ heatmap
                                                                                         └─▶ low-pollen route
```

1. **Plant gate** – CLIP zero-shot check rejects photos that are not plants.
2. **Close-up gate** – CLIP rejects distant or landscape shots and asks the user to retake a close-up of the foliage.
3. **Background removal** – BiRefNet segments the foliage and removes the background.
4. **Species classification** – a fine-tuned ResNet50 predicts `sugi`, `hinoki` or `other`. Predictions below a confidence threshold are rejected as "no match".
5. **Enrolment** – a confident sugi/hinoki detection is saved with its GPS location (from EXIF or given manually).
6. **Heatmap** – detections are weighted by confidence and rendered as a Folium heatmap.
7. **Routing** – each edge of the OpenStreetMap walking network gets a pollen cost from a Gaussian kernel around detections, and A* finds a route that trades a little extra distance for lower exposure.

## Repository structure

```
.
├── src/
│   └── model_utils.py              # ResNet18/50 builder (linear or dropout head)
├── scripts/
│   ├── data/
│   │   ├── download_images.py      # download images from a GBIF multimedia export
│   │   └── split_dataset.py        # split images/<class>/ into train/val
│   ├── training/
│   │   └── train.py                # transfer learning on ResNet50
│   ├── evaluation/
│   │   ├── evaluate.py             # per-class recall and confusion matrix
│   │   └── find_mislabeled.py      # collect misclassified images for label review
│   ├── prediction/
│   │   ├── predict.py              # classify a single image
│   │   ├── predict_gate.py         # full gated pipeline on a single image
│   │   └── test_pipeline.py        # end-to-end accuracy, including correct rejections
│   └── visualisation/
│       ├── enroll_detection.py     # classify a photo and add it to detections.json
│       ├── simulate_detections.py  # generate synthetic detections around OIC
│       ├── build_heatmap.py        # render heatmap.html
│       ├── pull_network.py         # download the OIC–JR Ibaraki walking network
│       └── route.py                # shortest vs low-pollen route
├── make.py                         # normalised confusion matrix for slides
├── viz_model.py                    # confusion matrix and per-class precision/recall plots
└── archive/                        # earlier training/evaluation versions
```

## Setup

Requires Python 3.10+. A CUDA GPU is strongly recommended for training and for the gated pipeline.

```bash
git clone https://github.com/Deoxyribonucleic/kafun.git
cd kafun
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision transformers pillow requests numpy matplotlib seaborn folium osmnx networkx shapely
```

Datasets, images and model checkpoints (`*.pth`) are not included in the repository.

## Usage

### 1. Prepare data

Download images per class from a [GBIF](https://www.gbif.org/) occurrence download (iNaturalist records by default), then split them.

```bash
python3 scripts/data/download_images.py --multimedia multimedia.txt --out images/sugi --limit 300
python3 scripts/data/split_dataset.py --src images --dst data --val_ratio 0.2
```

Expected layout:

```
data/
├── train/{sugi,hinoki,other}/
└── val/{sugi,hinoki,other}/
```

### 2. Train

```bash
python3 scripts/training/train.py --data_dir ./data --epochs 25 --batch_size 16
```

The best checkpoint by validation accuracy is saved to `kafun_model.pth`. Lower `--batch_size` to 8 if you run out of GPU memory.

### 3. Evaluate

```bash
python3 scripts/evaluation/evaluate.py --data_dir ./data --model kafun_model.pth
python3 scripts/evaluation/find_mislabeled.py --data_dir ./data --model kafun_model.pth --out review
```

`find_mislabeled.py` copies misclassified images into `review/<true>_AS_<pred>/`, prefixed with the model's confidence, so high-confidence mistakes (likely label errors) can be checked first.

### 4. Predict

```bash
python3 scripts/prediction/predict.py photo.jpg --thresh 0.7
python3 scripts/prediction/predict_gate.py photo.jpg --model kafun_model.pth
```

End-to-end test over `testset/{sugi,hinoki,other,notplant,far}/`:

```bash
python3 scripts/prediction/test_pipeline.py --testdir ./testset --model kafun_model.pth
```

### 5. Map and route

```bash
python3 scripts/visualisation/enroll_detection.py photo.jpg                       # GPS from EXIF
python3 scripts/visualisation/enroll_detection.py photo.jpg --lat 34.809 --lon 135.560
python3 scripts/visualisation/simulate_detections.py                              # synthetic data for testing
python3 scripts/visualisation/build_heatmap.py                                    # -> heatmap.html
python3 scripts/visualisation/pull_network.py                                     # -> oic_walk.graphml
python3 scripts/visualisation/route.py                                            # -> route.html
```

In `route.py`, `LAMBDA` sets how strongly pollen is avoided (0 gives the plain shortest path) and `BW_M` sets the influence radius of each detection in metres.

## Limitations

- Sugi and hinoki foliage look very similar, and web-sourced images vary widely in distance, angle and quality, so classification accuracy is the main bottleneck. Current work focuses on cleaning labels and enforcing close-up photos rather than increasing model size.
- The heatmap and routing demos use simulated detections until enough real submissions are collected.

## Author

Ryota Kawata, Ritsumeikan University
