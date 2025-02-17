import argparse
import glob
import os
from PIL import Image


def main(args):
    # Scale factor for x4 downscaling
    scale_list = [0.25]

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            #rlt.save(os.path.join(args.output, f'{basename}.jpg'), 'JPEG', quality=95)
            rlt.save(os.path.join(args.output, f'{basename}.jpg'), 'JPEG')


if __name__ == '__main__':
    """Generate x4 downscaled images using LANCZOS resampling."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='_CN/test_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='_CN/train_LR', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)