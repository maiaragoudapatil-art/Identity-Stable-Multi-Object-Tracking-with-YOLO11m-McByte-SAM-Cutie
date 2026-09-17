import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# ============================================================
# PROJECT ROOT
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# IMPORTANT:
# This file expects the McByte class to be available in:
#     yolox/tracker/mcbyte_tracker.py
#
# Do NOT paste this code into mcbyte_tracker.py.
from yolox.tracker.mcbyte_tracker import McByteTracker

# Your SAM + Cutie mask manager
from mask_propagation.mask_manager import MaskManager


# ============================================================
# SETTINGS
# ============================================================
DEFAULT_MODEL = "yolo11m.pt"

PERSON_CLASS = 0

YOLO_CONF = 0.20
YOLO_IOU = 0.50

TRACK_THRESH = 0.20
TRACK_BUFFER = 600

CMC_METHOD = "none"

MASK_ALPHA = 0.55
BOX_THICKNESS = 2
FONT_SCALE = 0.65


# ============================================================
# VIDEO PATH
# ============================================================
def get_video_path():
    while True:
        path = input("\nEnter full video path: ").strip().strip('"')

        if os.path.isfile(path):
            suffix = Path(path).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}:
                print("\nWARNING: This file does not look like a video:")
                print(path)
                continue
            return path

        print("\nERROR: Video does not exist:")
        print(path)


# ============================================================
# MODEL PATH
# ============================================================
def get_model_path():
    path = input(
        "\nEnter YOLO model path [Enter = yolo11m.pt]: "
    ).strip().strip('"')

    if not path:
        path = DEFAULT_MODEL

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "\nYOLO model not found:\n{}".format(path)
        )

    return path


# ============================================================
# FRAME RANGE
# ============================================================
def select_frames(total_frames):
    print("\n========== VIDEO FRAME SELECTION ==========")
    print("1. Process full video")
    print("2. Process selected frame range")

    while True:
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return 1, total_frames

        if choice == "2":
            try:
                start = int(input("Enter START frame number: "))
                end = int(input("Enter END frame number: "))

                if start < 1:
                    print("START must be >= 1")
                    continue

                if end < start:
                    print("END must be >= START")
                    continue

                end = min(end, total_frames)

                return start, end

            except ValueError:
                print("Enter valid integer frame numbers.")


# ============================================================
# MCByte ARGUMENTS
# ============================================================
def create_args(fps):
    args = SimpleNamespace()

    args.track_thresh = float(TRACK_THRESH)
    args.track_buffer = int(TRACK_BUFFER)
    args.cmc_method = CMC_METHOD
    args.fps = max(1, int(round(fps)))

    return args


# ============================================================
# STABLE COLOR FOR EACH TRACK ID
# ============================================================
def get_track_color(track_id):
    """
    Generates a deterministic bright BGR color.
    The same track ID always receives the same color.
    """

    hue = int((int(track_id) * 37) % 180)

    hsv = np.uint8(
        [[[hue, 220, 255]]]
    )

    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )[0, 0]

    return (
        int(bgr[0]),
        int(bgr[1]),
        int(bgr[2])
    )


# ============================================================
# COLOR SEGMENTATION MASKS
# ============================================================
def draw_colored_masks(
    frame,
    prediction_mask,
    tracklet_mask_dict,
    alpha=MASK_ALPHA
):
    """
    prediction_mask:
        Integer Cutie/SAM instance mask.

    tracklet_mask_dict:
        {track_id: mask_id}

    Only the pixels belonging to a known McByte track are colored.
    This keeps the segmentation colors tied to the tracking ID.
    """

    if prediction_mask is None:
        return frame

    if not isinstance(tracklet_mask_dict, dict):
        return frame

    mask = np.asarray(prediction_mask)

    if mask.ndim != 2:
        return frame

    h, w = frame.shape[:2]

    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.int32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    result = frame.copy()

    # Build the overlay separately.
    overlay = frame.copy()

    for track_id, mask_id in tracklet_mask_dict.items():

        try:
            tid = int(track_id)
            mid = int(mask_id)
        except (TypeError, ValueError):
            continue

        if mid <= 0:
            continue

        binary = mask == mid

        if not np.any(binary):
            continue

        color = get_track_color(tid)

        overlay[binary] = color

    return cv2.addWeighted(
        overlay,
        float(alpha),
        result,
        1.0 - float(alpha),
        0.0
    )


# ============================================================
# DRAW TRACK BOXES + IDS
# ============================================================
def draw_tracks(
    frame,
    online_targets,
    width,
    height
):
    active_ids = []

    for track in online_targets:

        if not hasattr(track, "track_id"):
            continue

        if not hasattr(track, "tlwh"):
            continue

        try:
            track_id = int(track.track_id)

            tlwh = np.asarray(
                track.tlwh,
                dtype=np.float32
            ).reshape(-1)

            if tlwh.size < 4:
                continue

            x, y, w, h = tlwh[:4]

            if not np.isfinite(
                [x, y, w, h]
            ).all():
                continue

            x1 = max(0, min(width - 1, int(round(x))))
            y1 = max(0, min(height - 1, int(round(y))))

            x2 = max(
                x1 + 1,
                min(width - 1, int(round(x + w)))
            )

            y2 = max(
                y1 + 1,
                min(height - 1, int(round(y + h)))
            )

        except Exception:
            continue

        color = get_track_color(track_id)

        active_ids.append(track_id)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS
        )

        # ID label background
        label = "ID {}".format(track_id)

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            2
        )

        label_y1 = max(
            0,
            y1 - th - baseline - 4
        )

        label_y2 = y1

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (min(width - 1, x1 + tw + 8), label_y2),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, max(th + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame, sorted(set(active_ids))


# ============================================================
# DRAW INFORMATION
# ============================================================
def draw_info(
    frame,
    frame_number,
    detections_count,
    active_ids
):
    cv2.putText(
        frame,
        "Frame: {}".format(frame_number),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "YOLO detections: {}".format(detections_count),
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Active IDs: {}".format(len(active_ids)),
        (15, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n==============================================")
    print(" YOLO11m + SAM + Cutie + McByte")
    print(" COLORED PERSON MASK + STABLE TRACK ID")
    print("==============================================")

    # --------------------------------------------------------
    # 1. YOLO MODEL
    # --------------------------------------------------------
    model_path = get_model_path()

    print("\nLoading YOLO11m...")
    print("YOLO model :", model_path)

    # Fail immediately if the model path is accidentally set to the video.
    # Ultralytics may otherwise defer this error until the first predict().
    model_suffix = Path(model_path).suffix.lower()
    allowed_model_suffixes = {
        ".pt", ".onnx", ".engine", ".torchscript", ".xml",
        ".tflite", ".pb", ".savedmodel", ".mlmodel", ".bin"
    }
    if model_suffix not in allowed_model_suffixes:
        raise ValueError(
            "\nThe YOLO MODEL path is not a model file:\n{}\n\n"
            "Enter the .pt YOLO model first, then enter the .mp4 video."
            .format(model_path)
        )

    model = YOLO(model_path)

    # Force model loading now, before video processing starts.
    _ = model.model

    print("YOLO11m loaded successfully:", model_path)

    # --------------------------------------------------------
    # 2. VIDEO
    # --------------------------------------------------------
    video_path = get_video_path()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video:\n{}".format(video_path)
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(
            "Invalid video dimensions: {}x{}".format(
                width,
                height
            )
        )

    print("\n========== VIDEO INFORMATION ==========")
    print("Video       :", video_path)
    print("Width       :", width)
    print("Height      :", height)
    print("FPS         :", fps)
    print("Total frames:", total_frames)

    # --------------------------------------------------------
    # 3. FRAME RANGE
    # --------------------------------------------------------
    start_frame, end_frame = select_frames(
        total_frames
    )

    print(
        "\nProcessing frames {} to {}".format(
            start_frame,
            end_frame
        )
    )

    # --------------------------------------------------------
    # 4. OUTPUT
    # --------------------------------------------------------
    timestamp = datetime.now().strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    output_folder = (
        ROOT
        / "YOLOX_outputs"
        / "yolo11m_mcbyte_colored"
        / timestamp
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_save_dir = (
        output_folder / "frames"
    )

    frames_save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    video_name = Path(
        video_path
    ).stem

    output_video = (
        output_folder
        / "{}_YOLO11m_McByte_SAM_Cutie_colored.mp4".format(
            video_name
        )
    )

    output_txt = (
        output_folder
        / "{}_YOLO11m_McByte_tracks.txt".format(
            video_name
        )
    )

    print("\nOutput video:")
    print(output_video)

    # --------------------------------------------------------
    # 5. VIDEO WRITER
    # --------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create output video:\n{}".format(
                output_video
            )
        )

    # --------------------------------------------------------
    # 6. MCByte
    # --------------------------------------------------------
    args = create_args(fps)

    print("\nCreating McByte tracker...")

    tracker = McByteTracker(
        args,
        save_folder=str(output_folder),
        frame_rate=args.fps
    )

    print("McByte tracker created.")

    # --------------------------------------------------------
    # 7. SAM / CUTIE
    # --------------------------------------------------------
    print("\nInitializing SAM/Cutie...")

    try:
        mask_manager = MaskManager()
    except Exception as exc:
        writer.release()
        cap.release()

        raise RuntimeError(
            "\nSAM/Cutie initialization failed.\n"
            "Tracking code is OK, but colored masks require "
            "a working MaskManager.\n\n"
            "Original error:\n{}".format(exc)
        ) from exc

    print("SAM/Cutie initialized.")

    # --------------------------------------------------------
    # 8. TRACK / MASK STATE
    # --------------------------------------------------------
    prediction_mask = None
    tracklet_mask_dict = {}
    mask_avg_prob_dict = {}

    previous_raw_frame = None

    previous_online_tlwhs = []
    previous_online_ids = []

    previous_new_tracks = []
    previous_removed_tracks_ids = []

    processed = 0

    # IMPORTANT:
    # MaskManager frame_id MUST start at 1 even if the user selects
    # a later video frame.
    local_frame_id = 0

    # --------------------------------------------------------
    # 9. SEEK
    # --------------------------------------------------------
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame - 1
    )

    results_file = open(
        output_txt,
        "w",
        encoding="utf-8"
    )

    try:

        while local_frame_id < (
            end_frame - start_frame + 1
        ):

            ret, raw_frame = cap.read()

            if not ret:
                print(
                    "\nCould not read frame."
                )
                break

            local_frame_id += 1

            frame_number = (
                start_frame
                + local_frame_id
                - 1
            )

            # ALWAYS keep an untouched copy.
            # Never give the annotated frame to SAM/Cutie.
            raw_frame = raw_frame.copy()

            img_info = {
                "height": height,
                "width": width,
                "raw_img": raw_frame
            }

            print(
                "\nFrame {} | local mask frame {}".format(
                    frame_number,
                    local_frame_id
                )
            )

            # ====================================================
            # 10. YOLO11 DETECTION
            # ====================================================
            result = model.predict(
                source=raw_frame,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                classes=[PERSON_CLASS],
                verbose=False
            )[0]

            detections = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                scores = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                for box, score in zip(
                    boxes,
                    scores
                ):

                    x1, y1, x2, y2 = (
                        box.astype(np.float32)
                    )

                    # Clamp detector coordinates.
                    x1 = max(
                        0.0,
                        min(float(width - 1), float(x1))
                    )

                    y1 = max(
                        0.0,
                        min(float(height - 1), float(y1))
                    )

                    x2 = max(
                        x1 + 1.0,
                        min(float(width), float(x2))
                    )

                    y2 = max(
                        y1 + 1.0,
                        min(float(height), float(y2))
                    )

                    detections.append(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                            float(score)
                        ]
                    )

            if detections:

                detection_array = np.asarray(
                    detections,
                    dtype=np.float32
                )

            else:

                detection_array = np.empty(
                    (0, 5),
                    dtype=np.float32
                )

            print(
                "YOLO11 detections:",
                len(detection_array)
            )

            # ====================================================
            # 11. SAM + CUTIE
            #
            # IMPORTANT ORDER:
            #
            # previous tracks
            #       ↓
            # SAM/Cutie propagation
            #       ↓
            # current McByte update
            #
            # This avoids the old argument/state mismatch.
            # ====================================================
            if (
                local_frame_id > 1
                and previous_raw_frame is not None
                and len(previous_online_ids) > 0
            ):

                previous_img_info = {
                    "height": height,
                    "width": width,
                    "raw_img": previous_raw_frame
                }

                try:

                    (
                        prediction_mask,
                        tracklet_mask_dict,
                        mask_avg_prob_dict,
                        _prediction_colors_preserved
                    ) = mask_manager.get_updated_masks(
                        img_info,
                        previous_img_info,
                        local_frame_id,
                        previous_online_tlwhs,
                        previous_online_ids,
                        previous_new_tracks,
                        previous_removed_tracks_ids
                    )

                    if tracklet_mask_dict is None:
                        tracklet_mask_dict = {}

                    if mask_avg_prob_dict is None:
                        mask_avg_prob_dict = {}

                except Exception as exc:

                    print(
                        "\n[WARNING] SAM/Cutie failed "
                        "on frame {}: {}".format(
                            frame_number,
                            exc
                        )
                    )

                    # Keep tracking alive.
                    prediction_mask = None
                    tracklet_mask_dict = {}
                    mask_avg_prob_dict = {}

            # ====================================================
            # 12. MCByte UPDATE
            # ====================================================
            (
                online_targets,
                removed_tracks_ids,
                new_tracks,
                _detections_per_assoc_step,
                _all_considered
            ) = tracker.update(
                detection_array,
                [height, width],
                [height, width],
                prediction_mask=prediction_mask,
                tracklet_mask_dict=tracklet_mask_dict,
                mask_avg_prob_dict=mask_avg_prob_dict,
                frame_img=raw_frame,
                vis_type="basic",
                dets_from_file=False
            )

            # ====================================================
            # 13. CURRENT TRACK STATE
            # ====================================================
            online_tlwhs = []
            online_ids = []

            for track in online_targets:

                if not hasattr(
                    track,
                    "track_id"
                ):
                    continue

                if not hasattr(
                    track,
                    "tlwh"
                ):
                    continue

                try:

                    tlwh = np.asarray(
                        track.tlwh,
                        dtype=np.float32
                    )

                    if tlwh.size < 4:
                        continue

                    track_id = int(
                        track.track_id
                    )

                    score = float(
                        getattr(
                            track,
                            "score",
                            0.0
                        )
                    )

                    online_tlwhs.append(
                        tlwh[:4]
                    )

                    online_ids.append(
                        track_id
                    )

                    results_file.write(
                        "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},-1,-1,-1\n".format(
                            frame_number,
                            track_id,
                            float(tlwh[0]),
                            float(tlwh[1]),
                            float(tlwh[2]),
                            float(tlwh[3]),
                            score
                        )
                    )

                except Exception:
                    continue

            # ====================================================
            # 14. DRAW COLORED SEGMENTATION
            # ====================================================
            annotated = raw_frame.copy()

            annotated = draw_colored_masks(
                annotated,
                prediction_mask,
                tracklet_mask_dict,
                MASK_ALPHA
            )

            # ====================================================
            # 15. DRAW TRACK BOXES + IDs
            # ====================================================
            annotated, active_ids = draw_tracks(
                annotated,
                online_targets,
                width,
                height
            )

            # ====================================================
            # 16. INFO
            # ====================================================
            annotated = draw_info(
                annotated,
                frame_number,
                len(detection_array),
                active_ids
            )

            # ====================================================
            # 17. SAVE FRAME + VIDEO
            # ====================================================
            frame_filename = (
                frames_save_dir
                / "frame_{:06d}.jpg".format(
                    frame_number
                )
            )

            cv2.imwrite(
                str(frame_filename),
                annotated
            )

            writer.write(
                annotated
            )

            processed += 1

            # ====================================================
            # 18. SAVE PREVIOUS RAW FRAME / TRACK STATE
            # ====================================================
            previous_raw_frame = raw_frame.copy()

            previous_online_tlwhs = [
                np.asarray(
                    x,
                    dtype=np.float32
                ).copy()
                for x in online_tlwhs
            ]

            previous_online_ids = [
                int(x)
                for x in online_ids
            ]

            previous_new_tracks = (
                list(new_tracks)
                if new_tracks is not None
                else []
            )

            previous_removed_tracks_ids = (
                list(removed_tracks_ids)
                if removed_tracks_ids is not None
                else []
            )

            print(
                "Active IDs:",
                sorted(set(active_ids))
            )

            # ----------------------------------------------------
            # Optional ESC / Q stop
            # ----------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("\nStopped by user.")
                break

    except KeyboardInterrupt:

        print(
            "\n\nProcessing interrupted."
        )

    finally:

        results_file.close()
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    # ============================================================
    # DONE
    # ============================================================
    print("\n==============================================")
    print("PROCESSING FINISHED")
    print("==============================================")
    print("Processed frames :", processed)
    print("Output video     :", output_video)
    print("Tracking results :", output_txt)
    print("Saved frames     :", frames_save_dir)
    print("==============================================")


if __name__ == "__main__":
    main()
