import argparse
import os
import os.path as osp
import sys
import time
import cv2
import torch
import numpy as np

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking, plot_tracking_basic
from yolox.tracker.mcbyte_tracker import McByteTracker

from mask_propagation.mask_manager import MaskManager

SAM_START_FRAME = 1

OVERLAP_MEASURE_VARIANT_1 = True
OVERLAP_MEASURE_VARIANT_2 = False
GRID_STEP = 10
MASK_CREATION_BB_OVERLAP_THRESHOLD = 0.6

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, possible types: image, video"
    )

    parser.add_argument(
        "--path", default=None, help="path to a folder with images (frames) or to a video file; if omitted for video, enter it in the terminal"
        # e.g. datasets/SoccerNet/tracking/test/SNMOT-132/img1/
    )
    parser.add_argument(
        "--det_path", 
        default=None, 
        # e.g. 'datasets/SoccerNet/tracking/test/SNMOT-132/det/det.txt'
        help="path to the file with detections; if specified, [detector] related arguments (e.g. exp_file, ckpt) will not be considered"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_x_mix_det.py",
        # default="exps/example/mot/yolox_x_ablation.py",
        type=str,
        help="[detector] your expriment description file",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="pretrained/yolox_x_sports_mix.pth.tar", 
        # default="pretrained/bytetrack_ablation.pth.tar", 
        type=str, 
        help="[detector] pretrained model weight (checkpoint) for eval"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, help="name of the parent output folder (default is the name of the exp_file)")
    parser.add_argument("--experiment_model_name", type=str, default=None, help="[detector] alternative to --exp_file")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="[detector] Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="[detector] Fuse conv and bn for testing.",
    )
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.001, help="track and detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="number of frames to keep lost tracks")

    # CMC
    parser.add_argument("--cmc-method", default="none", type=str, choices=["none", "orb", "ecc", "files"], help="camera motion compensation: none (recommended for fixed camera), orb, ecc, or files")

    parser.add_argument("--start_frame_no", type=int, default=None, help="starting frame number; interactive if omitted")
    parser.add_argument("--end_frame_no", type=int, default=None, help="ending frame number; interactive if omitted")
    parser.add_argument("--vis_type", default="basic", type=str, help="visualization type, with OR without detections and tracklets before Kalman filter update OR no visualization: full | basic | no_vis")

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = 0.001
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)

            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre
            )

            # Additional class-agnostic NMS to remove duplicate
            # boxes belonging to the same person.
            if outputs[0] is not None and len(outputs[0]) > 0:

                det = outputs[0].detach().cpu().numpy()

                boxes = det[:, :4]
                scores = det[:, 4] * det[:, 5]

                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                order = scores.argsort()[::-1]
                keep = []

                while len(order) > 0:

                    i = order[0]
                    keep.append(i)

                    if len(order) == 1:
                        break

                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])

                    w = np.maximum(0, xx2 - xx1)
                    h = np.maximum(0, yy2 - yy1)

                    intersection = w * h
                    union = areas[i] + areas[order[1:]] - intersection

                    iou = intersection / np.maximum(union, 1e-6)

                    order = order[1:][iou < 0.45]

                outputs[0] = outputs[0][keep]

        return outputs, img_info
            

def get_detections(img, frame_id, det_list):   
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    dets_per_frame = [d for d in det_list if d.split(",")[0] == str(frame_id)]

    dets_tensor = torch.zeros(len(dets_per_frame),5) # to be: x1 y1 x2 y2 conf

    for i, line in enumerate(dets_per_frame):
        det = line.split(",") # "frame_id,-1,left,top,width,height,conf,-1,-1,-1"

        dets_tensor[i,0] = int(det[2])
        dets_tensor[i,1] = int(det[3])
        dets_tensor[i,2] = int(det[4]) + int(det[2])
        dets_tensor[i,3] = int(det[5]) + int(det[3])
        dets_tensor[i,4] = int(det[6])

    # To adjust to the format used
    dets_tensor = dets_tensor[None, :]

    dets_array = dets_tensor.numpy()

    return dets_array, img_info


def image_demo(det_source, vis_folder, current_time, args, exp):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    files = files[args.start_frame_no-1:]

    ### For the info logging file save ###
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    ### / ###

    vis_type = args.vis_type
    if vis_type not in ['full', 'basic', 'no_vis']:
        print("[vis_type unrecognized, no visualization assumed]")

    if isinstance(det_source, Predictor):
        predictor = det_source
        dets_from_file = False
    elif isinstance(det_source, list):
        det_list = det_source
        dets_from_file = True
    else:
        print("[Unknown type of detection source, exiting.]")
        sys.exit()

    tracker = McByteTracker(args, frame_rate=args.fps, save_folder=save_folder)
    results = []

    # (These 2 lines + required indents) For Cutie
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

            prediction = None
            tracklet_mask_dict = {}
            mask_avg_prob_dict = {}
            prediction_colors_preserved = None

            mask_menager = MaskManager()

            for frame_id, img_path in enumerate(files, 1):
                print("Frame {}".format(str(frame_id)))
                if dets_from_file:
                    outputs, img_info = get_detections(img_path, frame_id + args.start_frame_no-1, det_list)
                else:
                    outputs, img_info = predictor.inference(img_path)
                
                if outputs[0] is not None:

                    if frame_id > 1:
                        prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved = mask_menager.get_updated_masks(img_info, img_info_prev, frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids)
                    
                    online_targets, removed_tracks_ids, new_tracks, detections_per_assoc_step, all_considered_tracklets_before_correction = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, prediction_mask=prediction, tracklet_mask_dict=tracklet_mask_dict, mask_avg_prob_dict=mask_avg_prob_dict, frame_img=img_info['raw_img'], vis_type=vis_type, dets_from_file=dets_from_file)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.last_det_tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                    if vis_type == 'full':
                        considered_online_tlwhs_before_correction = []
                        considered_online_ids_of_tracks_before_correction = []
                        for ct in all_considered_tracklets_before_correction:
                            considered_online_tlwhs_before_correction.append(ct.tlwh)
                            considered_online_ids_of_tracks_before_correction.append(ct.track_id)

                        online_im, online_im_dets, online_im_tracks_before_correction = plot_tracking(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, prediction_mask=prediction_colors_preserved, det_dict=detections_per_assoc_step, considered_online_tlwhs_before_correction=considered_online_tlwhs_before_correction, considered_online_ids_of_tracks_before_correction=considered_online_ids_of_tracks_before_correction
                        )
                    elif vis_type == 'basic':
                        online_im, online_im_dets, online_im_tracks_before_correction = plot_tracking_basic(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, prediction_mask=prediction_colors_preserved
                        )
                    else: # 'no_vis'
                        pass
                else:
                    online_im = img_info['raw_img']

                img_info_prev = img_info

                if args.save_result:
                    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_folder = osp.join(vis_folder, timestamp)
                    os.makedirs(save_folder, exist_ok=True)
                    if vis_type == 'full' or vis_type == 'basic':
                        cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
                    if vis_type == 'full':
                        cv2.imwrite(osp.join(save_folder, 'dets__' + osp.basename(img_path)), online_im_dets)
                        if not online_im_tracks_before_correction is None:
                            cv2.imwrite(osp.join(save_folder, 'tr_all__' + osp.basename(img_path)), online_im_tracks_before_correction)

                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def video_demo(det_source, vis_folder, current_time, args, exp):

    # ============================================================
    # 1. OPEN VIDEO
    # ============================================================

    if not args.path:
        args.path = input("Enter full video path: ").strip().strip('"').strip()

    if not args.path:
        raise ValueError("No video path was provided.")

    if not osp.isfile(args.path):
        raise FileNotFoundError(
            "Video file does not exist: {}".format(args.path)
        )

    cap = cv2.VideoCapture(args.path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video: {}".format(args.path)
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = float(args.fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n========== VIDEO INFORMATION ==========")
    print("Video       :", args.path)
    print("Width       :", width)
    print("Height      :", height)
    print("FPS         :", fps)
    print("Total frames:", total_frames)

    # ============================================================
    # 2. FRAME RANGE
    # ============================================================

    if args.start_frame_no is None or args.end_frame_no is None:

        print("\n========== VIDEO FRAME SELECTION ==========")
        print("1. Process full video")
        print("2. Process selected frame range")

        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":

            start_frame = 1
            end_frame = total_frames

        elif choice == "2":

            start_frame = int(
                input("Enter START frame number: ").strip()
            )

            end_frame = int(
                input("Enter END frame number: ").strip()
            )

        else:

            cap.release()
            raise ValueError("Please enter 1 or 2.")

    else:

        start_frame = args.start_frame_no
        end_frame = args.end_frame_no

    start_frame = max(1, start_frame)
    end_frame = min(total_frames, end_frame)

    if start_frame > end_frame:

        cap.release()

        raise ValueError(
            "START frame must be <= END frame."
        )

    # ============================================================
    # 3. OUTPUT FOLDER
    # ============================================================

    timestamp = time.strftime(
        "%Y_%m_%d_%H_%M_%S",
        current_time
    )

    save_folder = osp.join(
        vis_folder,
        timestamp
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    input_name = osp.splitext(
        osp.basename(args.path)
    )[0]

    save_path = osp.join(
        save_folder,
        "{}_frames_{}_to_{}.avi".format(
            input_name,
            start_frame,
            end_frame
        )
    )

    logger.info(
        "video save_path is {}".format(save_path)
    )

    # ============================================================
    # 4. VIDEO WRITER
    # ============================================================

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video: {}".format(save_path)
        )

    # ============================================================
    # 5. DETECTOR
    # ============================================================

    if isinstance(det_source, Predictor):

        predictor = det_source
        dets_from_file = False

    elif isinstance(det_source, list):

        det_list = det_source
        predictor = None
        dets_from_file = True

    else:

        cap.release()
        writer.release()

        raise RuntimeError(
            "Unknown detection source"
        )

    # ============================================================
    # 6. McByte TRACKER
    # ============================================================

    tracker = McByteTracker(
        args,
        frame_rate=fps,
        save_folder=save_folder
    )

    results = []

    # ============================================================
    # 7. MASK STATE
    #
    # IMPORTANT:
    # These must NEVER be None.
    # ============================================================

    prediction = None

    tracklet_mask_dict = {}

    mask_avg_prob_dict = {}

    prediction_colors_preserved = None

    # ============================================================
    # 8. MASK MANAGER
    # ============================================================

    # Initialize SAM/Cutie once. If segmentation dependencies/checkpoints are
    # unavailable, keep the tracker running with empty mask cues instead of
    # crashing the entire video job.
    try:
        mask_manager = MaskManager()
        mask_manager_enabled = True
        logger.info("SAM/Cutie mask manager initialized.")
    except Exception as e:
        mask_manager = None
        mask_manager_enabled = False
        logger.warning("SAM/Cutie initialization failed; continuing with YOLOX+McByte: {}".format(e))

    # Previous-frame information
    img_info_prev = None
    online_tlwhs = []
    online_ids = []

    # ============================================================
    # 9. MOVE VIDEO TO START FRAME
    # ============================================================

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame - 1
    )

    processed = 0

    # ============================================================
    # 10. MAIN LOOP
    # ============================================================

    try:

        with torch.inference_mode():

            for frame_no in range(
                start_frame,
                end_frame + 1
            ):

                ret, frame = cap.read()

                if not ret:

                    logger.warning(
                        "Could not read frame {}".format(
                            frame_no
                        )
                    )

                    break

                print(
                    "\nFrame {}".format(frame_no)
                )

                # ====================================================
                # YOLOX DETECTION
                # ====================================================

                if dets_from_file:

                    outputs, img_info = get_detections(
                        frame,
                        frame_no,
                        det_list
                    )

                else:

                    outputs, img_info = predictor.inference(
                        frame
                    )

                output0 = (
                    outputs[0]
                    if outputs is not None
                    else None
                )

                det_count = (
                    0
                    if output0 is None
                    else len(output0)
                )

                print(
                    "YOLOX detections: {}".format(
                        det_count
                    )
                )

                # ====================================================
                # MASK PROPAGATION
                #
                # Only run after the first frame.
                # ====================================================

                if (
                    mask_manager_enabled
                    and mask_manager is not None
                    and frame_no > start_frame
                    and img_info_prev is not None
                    and len(online_ids) > 0
                ):

                    try:

                        (
                            prediction,
                            tracklet_mask_dict,
                            mask_avg_prob_dict,
                            prediction_colors_preserved
                        ) = mask_manager.get_updated_masks(
                            img_info,
                            img_info_prev,
                            frame_no,
                            online_tlwhs,
                            online_ids,
                            new_tracks,
                            removed_tracks_ids
                        )

                    except Exception as e:

                        print(
                            "\n[WARNING] Mask propagation failed:"
                        )

                        print(e)

                        # Do NOT crash tracking
                        prediction = None
                        tracklet_mask_dict = {}
                        mask_avg_prob_dict = {}
                        prediction_colors_preserved = None

                # ====================================================
                # SAFE DEFAULTS
                # ====================================================

                if tracklet_mask_dict is None:

                    tracklet_mask_dict = {}

                if mask_avg_prob_dict is None:

                    mask_avg_prob_dict = {}

                # ====================================================
                # McByte UPDATE
                # ====================================================

                online_tlwhs = []
                online_ids = []

                removed_tracks_ids = []
                new_tracks = []

                # ALWAYS call the tracker, even when YOLOX has zero detections.
                # This is important: a no-detection frame must advance the
                # Kalman filter and mark tracks as lost instead of freezing them.
                if output0 is None or len(output0) == 0:
                    tracker_input = np.empty((0, 6), dtype=np.float32)
                else:
                    tracker_input = output0

                (
                    online_targets,
                    removed_tracks_ids,
                    new_tracks,
                    detections_per_assoc_step,
                    all_considered_tracklets_before_correction
                ) = tracker.update(
                    tracker_input,
                    [
                        img_info["height"],
                        img_info["width"]
                    ],
                    exp.test_size,
                    prediction_mask=prediction,
                    tracklet_mask_dict=tracklet_mask_dict,
                    mask_avg_prob_dict=mask_avg_prob_dict,
                    frame_img=img_info["raw_img"],
                    vis_type=args.vis_type,
                    dets_from_file=dets_from_file
                )

                # =================================================
                # GET CURRENT TRACKS
                # =================================================

                for t in online_targets:
                    tlwh = t.last_det_tlwh
                    tid = t.track_id
                    score = float(t.score)

                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    results.append(
                        "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},-1,-1,-1\n".format(
                            frame_no,
                            tid,
                            tlwh[0],
                            tlwh[1],
                            tlwh[2],
                            tlwh[3],
                            score
                        )
                    )

                print("Active IDs: {}".format(online_ids))

                # ====================================================
                # DRAW TRACKING
                # ====================================================

                if args.vis_type == "basic":

                    online_im, _, _ = plot_tracking_basic(

                        img_info["raw_img"],

                        online_tlwhs,

                        online_ids,

                        frame_id=frame_no,

                        prediction_mask=prediction_colors_preserved
                    )

                elif args.vis_type == "full":

                    considered_online_tlwhs_before_correction = []
                    considered_online_ids_of_tracks_before_correction = []

                    if all_considered_tracklets_before_correction is not None:

                        for ct in all_considered_tracklets_before_correction:

                            considered_online_tlwhs_before_correction.append(
                                ct.tlwh
                            )

                            considered_online_ids_of_tracks_before_correction.append(
                                ct.track_id
                            )

                    online_im, _, _ = plot_tracking(

                        img_info["raw_img"],

                        online_tlwhs,

                        online_ids,

                        frame_id=frame_no,

                        prediction_mask=prediction_colors_preserved,

                        det_dict=detections_per_assoc_step,

                        considered_online_tlwhs_before_correction=(
                            considered_online_tlwhs_before_correction
                        ),

                        considered_online_ids_of_tracks_before_correction=(
                            considered_online_ids_of_tracks_before_correction
                        )
                    )

                else:

                    online_im = img_info["raw_img"].copy()

                # ====================================================
                # WRITE FRAME
                # ====================================================

                writer.write(
                    online_im
                )

                processed += 1

                # ====================================================
                # SAVE PREVIOUS FRAME INFORMATION
                # ====================================================

                img_info_prev = img_info

                # ====================================================
                # PRINT CURRENT IDS
                # ====================================================

                print(
                    "Active IDs:",
                    online_ids
                )

    except KeyboardInterrupt:

        print(
            "\n\n[STOPPED] Processing interrupted by user."
        )

        print(
            "Saving all processed frames..."
        )

    finally:

        # ============================================================
        # ALWAYS RELEASE VIDEO
        # ============================================================

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    # ============================================================
    # 11. SAVE TRACKING RESULTS
    # ============================================================

    result_end_frame = (
        start_frame + processed - 1
    )

    result_file = osp.join(

        save_folder,

        "{}_frames_{}_to_{}.txt".format(

            input_name,

            start_frame,

            result_end_frame
        )
    )

    with open(
        result_file,
        "w"
    ) as f:

        f.writelines(results)

    # ============================================================
    # 12. FINAL INFORMATION
    # ============================================================

    logger.info(
        "save results to {}".format(
            result_file
        )
    )

    logger.info(
        "Output video: {}".format(
            save_path
        )
    )

    logger.info(
        "Processed {} frames".format(
            processed
        )
    )

    print("\n========================================")
    print("PROCESSING FINISHED")
    print("========================================")
    print("Processed frames :", processed)
    print("Output video     :", save_path)
    print("Tracking results :", result_file)
    print("========================================\n")
    
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Args: {}".format(args))

    det_path = args.det_path
    if det_path is not None:
        # load detections from a file
        with open(det_path, "r") as f_det:
            det_list = f_det.readlines()
        logger.info("Using detections from the file: {}".format(det_path))
        det_source = det_list
    else:
        # activate detector
        model = exp.get_model().to(device)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()  # to FP16

        predictor = Predictor(model, exp, device, args.fp16)
        det_source = predictor

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(det_source, vis_folder, current_time, args, exp)
    elif args.demo == "video":
        video_demo(det_source, vis_folder, current_time, args, exp)
    else:
        print("[No valid input mode selected (--demo=...). Tracking not performed. Available modes: image, video]")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.experiment_model_name)

    main(exp, args)
