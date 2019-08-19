
import sys

sys.path.append('build/')

import os
from libpydenseflow import TVL1FlowExtractor, TVL1WarpFlowExtractor
import numpy as np

class FlowExtractor(object):

    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)

    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        frame_size = frame_list[0].shape[:2]
        rst = self._et.extract_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
            for i in xrange(n_out):
                ret[2*i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size)
                ret[2*i+1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size)
        else:
            import cv2
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in xrange(n_out):
                ret[2*i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size), new_size)
                ret[2*i+1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size), new_size)

        return ret

def save_optical_flow(output_folder, flow_frames):
    try:
        os.mkdir(output_folder)
    except OSError:
        pass
    nframes = len(flow_frames) / 2
    for i in xrange(nframes):
        out_x = '{0}/x_{1:04d}.jpg'.format(output_folder, i+1)
        out_y = '{0}/y_{1:04d}.jpg'.format(output_folder, i+1)
        cv2.imwrite(out_x, flow_frames[2*i])
        cv2.imwrite(out_y, flow_frames[2*i+1])

if __name__ == "__main__":
    if len(sys.argv) < 3: # TODO! argparse
        print ("Missing arguments.\n"
               "Usage: \n"
               "    python tools/action_flow.py <INPUT VIDEO> <OUTPUT FOLDER>")
        sys.exit(-1)

    input_video = sys.argv[1]
    output_folder = sys.argv[2]

    import cv2
    if os.path.exists(input_video):
        frame_list = []
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()
        while ret:
            frame_list.append(frame)
            ret, frame = cap.read()
        f = FlowExtractor(dev_id=0)
        flow_frames = f.extract_flow(frame_list)
        save_optical_flow(output_folder, flow_frames)
