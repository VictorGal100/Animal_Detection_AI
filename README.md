# Animal\_Detection\_AI 
## Quickstart (Windows)

```powershell
# Clone & venv
git clone https://github.com/VictorGal100/Animal_Detection_AI.git
cd Animal_Detection_AI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# COCO val2017 (images + annotations)
md data\coco2017\images -ea 0
md data\coco2017\annotations -ea 0
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip
tar -xf val2017.zip -C data\coco2017\images
tar -xf annotations_trainval2017.zip -C data\coco2017

# Build a small animal-only sample set
python scripts\select_coco_animals.py --coco_root data\coco2017 --out docs\samples --n 25

# Run YOLOv8 on samples
python src\predict.py --src docs\samples --model yolov8s.pt --conf 0.20 --imgsz 640


