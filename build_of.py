__author__ = 'yjxiong'

import cv2
import os
from multiprocessing import Pool, current_process

out_path = ''


def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print '{} done'.format(vid_name)
    return file_list


def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = int(current._identity[0]) - 1
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = './build/extract_gpu -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o zip'.format(vid_path, flow_x_path, flow_y_path, image_path, dev_id)

    os.system(cmd)
    print '{} {} done'.format(vid_id, vid_name)
    return True


if __name__ == '__main__':
    out_path = '/data1/alex/anet/flow_tvl1'
    import glob
    vid_list = glob.glob('/home/alex/clips/*avi')
    print len(vid_list)
    pool = Pool(4)
    pool.map(run_optical_flow, zip(vid_list, xrange(len(vid_list))))
    #file_list = pool.map(dump_frames, vid_list)
    #all_file_list = [f for x in file_list for f in x]
    #open('anet_image_list_nov_17.txt','w').writelines('\n'.join(all_file_list))
    #for i,v in enumerate(vid_list):
    #	run_optical_flow(v, 0)
    #    print i
