import argparse
import sys
import os
import os.path as osp
import glob
from pipes import quote
from multiprocessing import Pool, current_process


def run_optical_flow(vid_item, dev_id=0):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % args.num_gpu

    cmd = osp.join(args.df_path, 'build/extract_nvflow') + \
        ' -v={} -o={} -a={} -s={} -b={} -d={} -w={} -h={}' \
        .format(
        quote(full_path),
        quote(out_full_path),
        args.flow_type,
        1,
        args.bound,
        dev_id, args.new_width, args.new_height)

    os.system(cmd)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--level', type=int, choices=[1, 2], default=1)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--bound', type=int, default=32)
    parser.add_argument('--flow_type', type=str, default='nv', choices=[None, 'nv', 'tvl1', 'warp_tvl1'])
    parser.add_argument('--df_path', type=str, default='/home/lizz/dev/dense_flow')
    parser.add_argument("--ext", type=str, default='webm', choices=['avi', 'mp4', 'webm'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
    parser.add_argument("--resume",
                        action='store_true',
                        default=False,
                        help='resume optical flow extraction '
                        'instead of overwriting')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    if args.level == 2:
        fullpath_list = glob.glob(args.src_dir + '/*/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(map(lambda p: osp.join('/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(run_optical_flow, zip(fullpath_list, vid_list, range(len(vid_list))))
