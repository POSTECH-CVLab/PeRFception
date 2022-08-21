import argparse
import json
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", default=".", type=str, help="path to the directory"
    )
    args = parser.parse_args()

    scenes = os.listdir(args.datadir)
    print(f"Total {len(scenes)} scenes from {args.datadir}")
    psnr, ssim, lpips = [], [], []

    results, missing = [], []
    for scene in scenes:
        json_file = os.path.join(args.datadir, scene, "results.json")
        if not os.path.exists(json_file):
            print(f"Not exist: {json_file}")
            missing.append(scene)
            continue
        else:
            with open(json_file, "r") as f:
                data = json.load(f)
            results.append(
                dict(
                    scene=scene,
                    psnr=data["PSNR"]["mean"],
                    ssim=data["SSIM"]["mean"],
                    lpips=data["LPIPS"]["mean"],
                )
            )

    score_name = ("psnr", "ssim", "lpips")

    for name in score_name:
        # print(f"{name} : {np.array(eval(name)).mean()}")
        metrics = np.array([r[name] for r in results])
        metrics = metrics[~np.isnan(metrics)]
        if name == "psnr":
            print(
                    f"{name:>5}: {np.mean(metrics):.3f}+-{np.std(metrics):.3f}, > 15: {np.mean(metrics > 15):.3f}, > 20: {np.mean(metrics > 20):.3f}, > 25: {np.mean(metrics > 25):.3f}, max: {np.max(metrics):.3f}, min: {np.min(metrics):.3f}, 25th percentile: {np.percentile(metrics, 25):.3f}, 50th percentile: {np.percentile(metrics, 50):.3f}, 75th percentile: {np.percentile(metrics, 75):.3f}, 90th: {np.percentile(metrics, 90):.3f}, 95th: {np.percentile(metrics, 95):.3f}"
            )
        else:
            print(
                f"avg {name:>5}: {np.mean(metrics):.3f}, max: {np.max(metrics):.3f}, min: {np.min(metrics):.3f}"
            )
