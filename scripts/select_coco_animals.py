# Alexander Garcia — copy N animal images from COCO val2017 to docs/samples
import json, random, shutil
from pathlib import Path
import argparse

ANIMALS = {"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", required=True)
    p.add_argument("--out", default="docs/samples")
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()

    root = Path(args.coco_root)
    ann = json.load(open(root/"annotations/instances_val2017.json"))
    img_dir = root/"images/val2017"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    name2id = {c["name"]: c["id"] for c in ann["categories"]}
    animal_ids = {name2id[n] for n in ANIMALS if n in name2id}
    animal_img_ids = {a["image_id"] for a in ann["annotations"] if a["category_id"] in animal_ids}

    pick = random.sample(list(animal_img_ids), k=min(args.n, len(animal_img_ids)))
    id2file = {i["id"]: i["file_name"] for i in ann["images"]}
    for iid in pick:
        src = img_dir/id2file[iid]
        if src.exists():
            shutil.copy(src, out/src.name)
    print(f"Copied {len(pick)} animal images to {out}")

if __name__ == "__main__":
    main()
