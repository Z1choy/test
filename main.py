import argparse
import os
from patchcore_runner import PatchCore
from data.mvtec3d import mvtec3d_classes
from data.real3d import real3d_classes
from data.anomalyshape import shapenet3d_classes
from data.MulSen import mulsen_classes
from data.MiniShiftAD import minishiftAD_classes
import pandas as pd
import torchvision


def write_experiment_log(expname, strs):
    os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{expname}.txt", 'a') as f:
        f.write(strs)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def run_3d_ads(args):
    

    if args.dataset == 'mvtec':
        classes = mvtec3d_classes()
    if args.dataset == 'real':
        classes = real3d_classes()
    if args.dataset == 'shapenet':
        classes = shapenet3d_classes()
    if args.dataset == 'eyecandies':
        classes = eyecandies_classes() 
    if args.dataset == 'mulsen':
        classes = mulsen_classes() 
    if args.dataset == 'minishift':
        classes = minishiftAD_classes() 
    if 'classes' not in locals():
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.category is not None:
        if args.category not in classes:
            raise ValueError(f"Category {args.category} is not in {args.dataset} classes: {classes}")
        classes = [args.category]
    METHOD_NAMES = [
        "Simple3D",
        ]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for category in classes:
        patchcore = PatchCore(args=args)
        # patchcore.train(category)
        patchcore.fit(category)

    # for cls in args.category:
        cls = category
        print(f"\nRunning on class {cls}\n")
        write_experiment_log(args.expname,f"\nRunning on class {cls}\n")
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

        print(f"\nFinished running on class {cls}\n")
        write_experiment_log(args.expname,f"\nFinished running on class {cls}\n")
        print("################################################################################\n\n")
        # write_experiment_log(args.expname,"################################################################################\n\n")
        

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--expname', type=str, default='None', help='expname')
    parser.add_argument('--device', type=str, default='cuda:0', help='expname')
    parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
    parser.add_argument('--max_nn', type=int, default=100, help='max_nn')
    parser.add_argument('--num_group', type=int, default=2048, help='num_group')
    parser.add_argument('--group_size', type=int, default=128, help='group_size')
    parser.add_argument('--use_MSND', type=str2bool, default=False)
    parser.add_argument('--use_LFSA', type=str2bool, default=False)
    parser.add_argument('--vis_save', type=str2bool, default=False)
    parser.add_argument('--num_MSND', type=int, default=2)
    parser.add_argument('--feature', type=str, default='FPFH')
    parser.add_argument('--level', type=str, default='ALL')
    parser.add_argument('--category', type=str, default=None)
    args = parser.parse_args()
    print(args)

    run_3d_ads(args)
