import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
from PIL import Image

# 3dmm extraction
import mindspore as ms
from models.face3d.utils import load_lm3d, align_img
from models.face3d.networks import define_net_recon

from scipy.io import savemat, loadmat
from utils.croper import Preprocesser


import warnings

warnings.filterwarnings("ignore")


def split_coeff(coeffs, return_dict=False):
    """
    Return:
        coeffs_dict     -- a dict of ms.tensors

    Parameters:
        coeffs          -- ms.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80:144]
    tex_coeffs = coeffs[:, 144:224]
    angles = coeffs[:, 224:227]
    gammas = coeffs[:, 227:254]
    translations = coeffs[:, 254:]

    if return_dict:
        return {
            "id": id_coeffs,
            "exp": exp_coeffs,
            "tex": tex_coeffs,
            "angle": angles,
            "gamma": gammas,
            "trans": translations,
        }
    else:
        return (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)


class CropAndExtract:
    def __init__(self, config):
        self.propress = Preprocesser()
        self.net_recon = define_net_recon(
            net_recon="resnet50", use_last_fc=False, init_path=""
        )

        checkpoint_dir = config.path.checkpoint_dir
        path_net_recon = os.path.join(
            checkpoint_dir, config.path.path_of_net_recon_model
        )
        path_bfm = os.path.join(checkpoint_dir, config.path.dir_of_bfm_fitting)

        param_dict = ms.load_checkpoint(path_net_recon)
        ms.load_param_into_net(self.net_recon, param_dict)
        self.net_recon.set_train(False)

        self.lm3d_std = load_lm3d(path_bfm)

    def generate(
        self,
        input_path,
        save_dir,
        crop_or_resize="crop",
        source_image_flag=False,
        pic_size=256,
    ):
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]

        landmarks_path = os.path.join(save_dir, pic_name + "_landmarks.txt")
        coeff_path = os.path.join(save_dir, pic_name + ".mat")
        png_path = os.path.join(save_dir, pic_name + ".png")

        # load input
        if not os.path.isfile(input_path):
            raise ValueError("input_path must be a valid path to video/image file")
        elif input_path.split(".")[-1] in ["jpg", "png", "jpeg"]:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                full_frames.append(frame)
                if source_image_flag:
                    break

        x_full_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames
        ]

        print("start cropping the image ...")

        # crop images as the
        if "crop" in crop_or_resize.lower():  # default crop
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif "full" in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else:  # resize mode
            oy1, oy2, ox1, ox2 = (
                0,
                x_full_frames[0].shape[0],
                0,
                x_full_frames[0].shape[1],
            )
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [
            Image.fromarray(cv2.resize(frame, (pic_size, pic_size)))
            for frame in x_full_frames
        ]
        if len(frames_pil) == 0:
            print("No face is detected in the input file")
            return None, None

        print("finished cropping, now saving the image to file.")

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        print(f"finished cropping the image and saved to file {png_path}.")

        # 2. get the landmark according to the detected face.
        if not os.path.isfile(landmarks_path):
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(" Using saved landmarks.")
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch
            video_coeffs, full_coeffs = [], []
            for idx in tqdm(range(len(frames_pil)), desc="3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm1 = lm[idx].reshape([-1, 2])

                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2] + 1) / 2.0
                    lm1 = np.concatenate([lm1[:, :1] * W, lm1[:, 1:2] * H], 1)
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)

                trans_params = np.array(
                    [float(item) for item in np.hsplit(trans_params, 5)]
                ).astype(np.float32)
                im_t = (
                    ms.Tensor(np.array(im1) / 255.0, dtype=ms.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )

                full_coeff = self.net_recon(im_t)
                coeffs = split_coeff(full_coeff)

                pred_coeff = [coef.asnumpy() for coef in coeffs]
                # (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)

                pred_coeff = np.concatenate(
                    [
                        pred_coeff[1],
                        pred_coeff[3],
                        pred_coeff[5],
                        trans_params[2:][None],
                    ],
                    1,
                )
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.asnumpy())

            semantic_npy = np.array(video_coeffs)[:, 0]

            savemat(
                coeff_path,
                {"coeff_3dmm": semantic_npy, "full_3dmm": np.array(full_coeffs)[0]},
            )

        return coeff_path, png_path, crop_info

    def generate_for_train(
        self,
        video_path,
        save_dir,
        org_dir,
        crop_or_resize="crop",
        source_image_flag=False,
        pic_size=256,
        fps=25,
    ):
        video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
        dirname = os.path.dirname(video_path)
        save_path = dirname.replace(org_dir, save_dir) + "/" + video_name
        os.makedirs(save_path, exist_ok=True)

        landmarks_path = os.path.join(
            save_path, video_name + "_landmarks.txt"
        )  # 每一帧的图像landmark
        img_path = save_path.replace("/pose/", "/images/")  # 最终的图像保存路径
        os.makedirs(img_path, exist_ok=True)  # 图像保存路径
        # info_path =  os.path.join(save_path, 'crop_info.mat')

        coeff_path = os.path.join(save_path, video_name + ".mat")

        # load input
        if not os.path.isfile(video_path):
            raise ValueError("input_path must be a valid path to video/image file")
        video_stream = cv2.VideoCapture(video_path)
        video_stream.set(cv2.CAP_PROP_FPS, fps)
        x_full_frames = []
        if video_stream.isOpened() == False:
            print("Error opening video stream:", video_path)
        else:
            success = True
            k = 0
            while success:
                success, frame = video_stream.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
                    x_full_frames.append(frame)
        print("读取视频完毕====>", video_path, "  ", os.getpid())

        #### crop images as the
        if "crop" in crop_or_resize.lower():  # default crop
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif "full" in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else:  # resize mode
            oy1, oy2, ox1, ox2 = (
                0,
                x_full_frames[0].shape[0],
                0,
                x_full_frames[0].shape[1],
            )
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [
            Image.fromarray(cv2.resize(frame, (pic_size, pic_size)))
            for frame in x_full_frames
        ]  # resize到256
        if len(frames_pil) == 0:
            print("No face is detected in the input file")
            return None, None

        # save crop info  保存crop的图像
        i = 1
        for frame in frames_pil:
            png_path = os.path.join(
                img_path, video_name + "_" + str(i).zfill(6) + ".png"
            )
            cv2.imwrite(
                png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            )  # 保存crop的图像
            i += 1

        # 2. get the landmark according to the detected face.
        if not os.path.isfile(landmarks_path):
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(" Using saved landmarks.")
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch
            video_coeffs, full_coeffs = [], []
            for idx in tqdm(range(len(frames_pil)), desc="3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm1 = lm[idx].reshape([-1, 2])

                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2] + 1) / 2.0
                    lm1 = np.concatenate([lm1[:, :1] * W, lm1[:, 1:2] * H], 1)
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)

                trans_params = np.array(
                    [float(item) for item in np.hsplit(trans_params, 5)]
                ).astype(np.float32)
                im_t = (
                    ms.tensor(np.array(im1) / 255.0, dtype=ms.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )

                full_coeff = self.net_recon(im_t)  # 1 257
                coeffs = split_coeff(full_coeff, return_dict=True)

                pred_coeff = {key: coeffs[key].asnumpy() for key in coeffs}

                pred_coeff = np.concatenate(
                    [
                        pred_coeff["exp"],
                        pred_coeff["angle"],
                        pred_coeff["trans"],
                        trans_params[2:][None],  # 仿射变换参数
                    ],
                    1,
                )  # (1, 73)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.asnumpy())

            semantic_npy = np.array(video_coeffs)[:, 0]

            savemat(
                coeff_path,
                {"coeff_3dmm": semantic_npy, "full_3dmm": np.array(full_coeffs)[0]},
            )

    # 检测landmark
    def detect_lm(
        self,
        input_path,
        save_dir,
        org_dir,
        crop_or_resize="crop",
        source_image_flag=False,
    ):
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        dirname = os.path.dirname(input_path)
        save_path = dirname.replace(org_dir, save_dir) + "/" + pic_name
        os.makedirs(save_path, exist_ok=True)
        landmarks_path = os.path.join(save_path, "first_landmarks.mat")

        # load input
        video_stream = cv2.VideoCapture(input_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
            if source_image_flag:
                break
            break  # 只读取第一帧

        x_full_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames
        ]

        ####保存第一帧的landmarks
        if "crop" in crop_or_resize.lower():  # default crop
            lm = self.propress.detect_lm(
                x_full_frames,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
        savemat(landmarks_path, {"first_lm": lm})

    # crop 256 and save
    def crop_256_save(
        self,
        input_path,
        save_dir,
        org_dir,
        crop_or_resize="crop",
        source_image_flag=False,
        pic_size=256,
    ):
        # print(input_path)
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        dirname = os.path.dirname(input_path)
        save_path = dirname.replace(org_dir, save_dir) + "/" + pic_name
        os.makedirs(save_path, exist_ok=True)
        landmarks_path = os.path.join(save_path, "first_landmarks.mat")
        if not os.path.exists(landmarks_path):  # 不存在return
            # print('不存在:', input_path)
            return
        img_path = save_path.replace("/pose/", "/images/")  # 图像保存路径
        os.makedirs(img_path, exist_ok=True)  # 图像保存路径
        info_path = os.path.join(save_path, "crop_info.mat")
        # if os.path.exists(info_path):
        #     return   #不需要了
        print("正在处理====>", input_path, "  ", os.getpid())
        # load input
        video_stream = cv2.VideoCapture(input_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        x_full_frames = []
        if video_stream.isOpened() == False:
            print("Error opening video stream:", input_path)
        else:
            success = True
            k = 0
            while success:
                success, frame = video_stream.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
                    x_full_frames.append(frame)
                    k += 1
        print("读取视频完毕====>", input_path, "  ", os.getpid())
        png_path = os.path.join(img_path, pic_name + "_" + str(k).zfill(6) + ".png")
        if os.path.exists(png_path):
            return
        print("不完整====>", input_path, "  ", os.getpid())

        ####align
        lm_68 = loadmat(landmarks_path)["first_lm"]
        if "crop" in crop_or_resize.lower():  # default crop
            x_full_frames, crop, quad = self.propress.align(
                x_full_frames,
                lm_68,
                still=True if "ext" in crop_or_resize.lower() else False,
                xsize=512,
            )
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        print("align视频完毕====>", input_path, "  ", os.getpid())

        frames_pil = [
            Image.fromarray(cv2.resize(frame, (pic_size, pic_size)))
            for frame in x_full_frames
        ]  # resize到256
        if len(frames_pil) == 0:
            print("No face is detected in the input file")
            return None, None

        # 保存crop的图像
        i = 1
        for frame in frames_pil:
            png_path = os.path.join(img_path, pic_name + "_" + str(i).zfill(6) + ".png")
            cv2.imwrite(
                png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            )  # 保存crop的图像
            i += 1
        savemat(info_path, {"crop_info": crop_info})
        print("保存视频完毕====>", input_path, "  ", os.getpid())

    def align_256_final(
        self,
        input_path,
        save_dir,
        org_dir,
        crop_or_resize="crop",
        source_image_flag=False,
        pic_size=256,
    ):
        # print(input_path)
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        dirname = os.path.dirname(input_path)
        save_path = dirname.replace(org_dir, save_dir) + "/" + pic_name
        os.makedirs(save_path, exist_ok=True)
        landmarks_path = os.path.join(save_path, "first_landmarks.mat")
        if not os.path.exists(landmarks_path):  # 不存在return
            # print('不存在:', input_path)
            return
        info_path = os.path.join(save_path, "crop_info.mat")
        print("正在处理====>", input_path, "  ", os.getpid())

        coeff_path = os.path.join(save_path, pic_name + ".mat")  # 3d系数保存路径
        if os.path.exists(coeff_path):  # 存在的话就return
            return

        img_path = save_path.replace("/pose/", "/images/")  # 图像保存路径
        os.makedirs(img_path, exist_ok=True)  # 图像保存路径
        img_paths = glob(os.path.join(img_path, "*.png"))  # 得到路径下的所有图像
        frames_pil = [Image.open(p) for p in img_paths]  # 打开

        landmarks_paths_256 = os.path.join(
            save_path, pic_name + "_landmarks.txt"
        )  # 256图像对应的landmarks
        if not os.path.isfile(landmarks_paths_256):
            lm = self.propress.predictor.extract_keypoint(
                frames_pil, landmarks_paths_256
            )
        else:
            print(" Using saved landmarks.")
            lm = np.loadtxt(landmarks_paths_256).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch
            video_coeffs, full_coeffs = [], []
            for idx in tqdm(range(len(frames_pil)), desc="3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm1 = lm[idx].reshape([-1, 2])

                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2] + 1) / 2.0
                    lm1 = np.concatenate([lm1[:, :1] * W, lm1[:, 1:2] * H], 1)
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)

                trans_params = np.array(
                    [float(item) for item in np.hsplit(trans_params, 5)]
                ).astype(np.float32)
                im_t = (
                    ms.tensor(np.array(im1) / 255.0, dtype=ms.float32)
                    .permute(2, 0, 1)
                    .to(self.device)
                    .unsqueeze(0)
                )

                full_coeff = self.net_recon(im_t)  # 1 257
                coeffs = split_coeff(full_coeff, return_dict=True)

                pred_coeff = {key: coeffs[key].asnumpy() for key in coeffs}

                pred_coeff = np.concatenate(
                    [
                        pred_coeff["exp"],
                        pred_coeff["angle"],
                        pred_coeff["trans"],
                        trans_params[2:][None],  # 仿射变换参数
                    ],
                    1,
                )  # (1, 73)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.asnumpy())

            semantic_npy = np.array(video_coeffs)[:, 0]

            savemat(
                coeff_path,
                {"coeff_3dmm": semantic_npy, "full_3dmm": np.array(full_coeffs)[0]},
            )
