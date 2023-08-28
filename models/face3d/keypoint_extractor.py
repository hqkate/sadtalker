import mindspore as ms
from mindspore import nn
import numpy as np
import tqdm
import time
import os

from models.face3d.facexlib import landmark_98_to_68
from models.face3d.facexlib import init_detection_model, init_alignment_model


class KeypointExtractor():
    def __init__(self, detector, det_net):

        self.detector = detector
        self.det_net = det_net

    def extract_keypoint(self, images, name=None, info=True):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images, desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                # current_kp = self.detector.get_landmarks(np.array(image))
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    # face detection -> face alignment.
                    img = np.array(images)
                    bboxes = self.det_net.detect_faces(images, 0.97)

                    bboxes = bboxes[0]
                    img = img[int(bboxes[1]):int(bboxes[3]),
                              int(bboxes[0]):int(bboxes[2]), :]

                    keypoints = landmark_98_to_68(
                        self.detector.get_landmarks(img))  # [0]

                    # keypoints to the original location
                    keypoints[:, 0] += int(bboxes[0])
                    keypoints[:, 1] += int(bboxes[1])

                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[
                           0]+'.txt', keypoints.reshape(-1))
            return keypoints


if __name__ == "__main__":
    # gfpgan/weights
    root_path = 'gfpgan/weights'
    detector = init_alignment_model(
        'awing_fan', model_rootpath=root_path)
    det_net = init_detection_model(
        'retinaface_resnet50', half=False, model_rootpath=root_path)
