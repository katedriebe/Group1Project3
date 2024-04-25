import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from util import ProtestDatasetEval, modified_resnet50

def eval_one_dir(img_dir, model, args):
    model.eval()
    dataset = ProtestDatasetEval(img_dir=img_dir)
    data_loader = DataLoader(dataset, num_workers=args.workers, batch_size=args.batch_size)

    outputs = []
    imgpaths = []

    n_imgs = len(os.listdir(img_dir))
    with torch.no_grad(), tqdm(total=n_imgs) as pbar:
        for i, sample in enumerate(data_loader):
            imgpath, input = sample['imgpath'], sample['image'].to(args.device)

            output = model(input)
            outputs.append(output.cpu().numpy())
            imgpaths.extend(imgpath)

            pbar.update(len(input))

    df = pd.DataFrame(np.zeros((len(imgpaths), 13)), columns=["imgpath", "protest", "violence", "sign", "photo",
                                                              "fire", "police", "children", "group_20", "group_100",
                                                              "flag", "night", "shouting"])
    df['imgpath'] = imgpaths
    df.iloc[:, 1:] = np.concatenate(outputs)
    df.sort_values(by='imgpath', inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate image directory with model")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--output_csvpath", type=str, default="result.csv", help="Path to output CSV file")
    parser.add_argument("--model_best", type=str, required=True, help="Path to the best model file")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading")
    args = parser.parse_args()

    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(f"*** Loading model from {args.model_best}")
    model = modified_resnet50().to(args.device)
    model.load_state_dict(torch.load(args.model_best, map_location=args.device)['state_dict'])

    print(f"*** Calculating the model output of the images in {args.img_dir}")
    df = eval_one_dir(args.img_dir, model, args)

    df.to_csv(args.output_csvpath, index=False)
    print(f"*** Results written to {args.output_csvpath}")

if __name__ == "__main__":
    main()
