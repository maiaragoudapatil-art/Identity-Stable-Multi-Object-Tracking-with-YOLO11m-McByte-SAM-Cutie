import torch
import cv2
import numpy as np
import sys
from pathlib import Path

### SAM ###
from segment_anything import sam_model_registry, SamPredictor

# Enable imports from the Cutie directory
root = Path(__file__).parent
_CUTIE_ROOT = root / 'Cutie'
sys.path.append(str(_CUTIE_ROOT))

# Hydra config path (relative)
cutie_config_rel = "Cutie/cutie/config"

### Cutie ###
from omegaconf import open_dict
from hydra import compose, initialize

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from gui.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
)

OVERLAP_MEASURE_VARIANT = 1
OVERLAP_VARIANT_2_GRID_STEP = 10
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6

# ============================================================
# PERFORMANCE
# ============================================================
CUTIE_MAX_SIDE = 960


class MaskManager(object):
    """
    ============================================================
    ID SCHEME (READ THIS BEFORE TOUCHING THE ADD/REMOVE LOGIC)
    ============================================================
    There is exactly ONE identifier space for a mask/object:
    `stable_id` (a permanently-unique integer, assigned by
    `self.next_stable_id`, NEVER reused, NEVER renumbered).

    stable_id is used, unchanged, as:
      - the pixel value painted into every mask array
        (mask, mask_extra, mask_prediction_prev_frame_cutie)
      - the value stored in `tracklet_mask_dict[track_id]`
      - the entries of `current_object_list_cutie`, which is the
        exact list passed as `objects=` to processor.step()

    `current_object_list_cutie` doubles as the POSITION MAP: the
    i-th entry (0-indexed) of this list is, by construction, the
    stable_id occupying prediction-tensor channel (i+1). This is
    the ONLY place position<->stable_id is derived, and it is
    rebuilt/kept in lockstep every time objects are added or
    removed -- never inferred separately, never decremented.

    Previously this codebase used THREE divergent counters
    (a "compact" counter for tracklet_mask_dict, a separate
    ever-increasing `last_object_number_cutie` for the objects=
    list, and a third `mask_color_counter` for the color remap).
    They drifted apart as soon as any track was removed, which
    could make a brand-new mask silently collide with a live
    object's internal id -- this was the source of ID/mask
    identity swaps at occlusion. Do not reintroduce a second
    counter; everything must derive from stable_id.
    """

    def __init__(self):
        self.masks = None
        self.mask = None
        self.prediction = None

        # track_id -> stable_id
        self.tracklet_mask_dict = {}

        # Position i (0-indexed) -> stable_id occupying that
        # channel in the Cutie prediction tensor / one-hot input.
        # len(current_object_list_cutie) == self.num_objects
        self.current_object_list_cutie = []

        # Single source of truth for new ids. Never reused.
        self.next_stable_id = 0

        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0

        self.num_objects = 0

        # IMPORTANT:
        # This is kept in ORIGINAL video resolution for output.
        self.mask_prediction_prev_frame = None

        # This is kept in CUTIE resolution for Cutie memory.
        self.mask_prediction_prev_frame_cutie = None

        self.SAM_START_FRAME = 1

        # Cutie frame geometry
        self.original_height = None
        self.original_width = None
        self.cutie_height = None
        self.cutie_width = None
        self.cutie_scale = 1.0

        # --------------------------------------------------------
        # SAM
        # --------------------------------------------------------
        np.random.seed(0)

        sam_checkpoint = (
            r"C:\Users\VijaySegunasi\McByteProject\McByte"
            r"\sam_models\sam_vit_b_01ec64.pth"
        )

        model_type = "vit_b"

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.sam = sam_model_registry[model_type](
            checkpoint=sam_checkpoint
        )

        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # --------------------------------------------------------
        # CUTIE
        # --------------------------------------------------------
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):
                initialize(
                    version_base='1.3.2',
                    config_path=str(cutie_config_rel),
                    job_name="eval_config"
                )

                cfg = compose(
                    config_name="eval_config"
                )

                weight_path = (
                    Path(__file__).parent
                    / "Cutie"
                    / "weights"
                    / "cutie-base-mega.pth"
                )

                with open_dict(cfg):
                    cfg['weights'] = weight_path

                _ = get_dataset_cfg(cfg)

                self.cutie = CUTIE(
                    cfg
                ).to(
                    self.device
                ).eval()

                model_weights = torch.load(
                    cfg.weights,
                    map_location=self.device
                )

                self.cutie.load_weights(
                    model_weights
                )

                torch.cuda.empty_cache()

                self.processor = InferenceCore(
                    self.cutie,
                    cfg=cfg
                )

    # ============================================================
    # CUTIE RESOLUTION HELPERS
    # ============================================================

    def _prepare_cutie_geometry(self, frame):
        h, w = frame.shape[:2]

        if self.original_height == h and self.original_width == w:
            return

        self.original_height = h
        self.original_width = w

        self.cutie_scale = min(
            1.0,
            float(CUTIE_MAX_SIDE) / float(max(h, w))
        )

        self.cutie_width = max(1, int(round(w * self.cutie_scale)))
        self.cutie_height = max(1, int(round(h * self.cutie_scale)))

    def _resize_frame_for_cutie(self, frame):
        self._prepare_cutie_geometry(frame)

        if (
            self.cutie_width == self.original_width
            and self.cutie_height == self.original_height
        ):
            return frame

        return cv2.resize(
            frame,
            (self.cutie_width, self.cutie_height),
            interpolation=cv2.INTER_AREA
        )

    def _resize_label_mask_for_cutie(self, mask):
        if mask is None:
            return None

        if (
            mask.shape[1] == self.cutie_width
            and mask.shape[0] == self.cutie_height
        ):
            return mask.astype(np.int32, copy=True)

        return cv2.resize(
            mask.astype(np.int32),
            (self.cutie_width, self.cutie_height),
            interpolation=cv2.INTER_NEAREST
        )

    def _resize_prediction_to_original(self, prediction):
        if prediction is None:
            return None

        cutie_mask = torch_prob_to_numpy_mask(prediction)

        if (
            cutie_mask.shape[0] == self.original_height
            and cutie_mask.shape[1] == self.original_width
        ):
            return cutie_mask

        return cv2.resize(
            cutie_mask.astype(np.int32),
            (self.original_width, self.original_height),
            interpolation=cv2.INTER_NEAREST
        )

    # ============================================================
    # ID HELPERS  (the fix lives here)
    # ============================================================

    def _new_stable_id(self):
        self.next_stable_id += 1
        return self.next_stable_id

    def _position_to_stable_map(self):
        """
        1-indexed prediction/one-hot CHANNEL -> stable_id, derived
        fresh from current_object_list_cutie every time. This is
        the single place that ever translates position<->identity.
        """
        return {
            pos + 1: stable_id
            for pos, stable_id in enumerate(self.current_object_list_cutie)
        }

    def _stable_to_position_map(self):
        return {
            stable_id: pos + 1
            for pos, stable_id in enumerate(self.current_object_list_cutie)
        }

    # ============================================================
    # MAIN MASK UPDATE
    # ============================================================

    def get_updated_masks(
        self,
        img_info,
        img_info_prev,
        frame_id,
        online_tlwhs,
        online_ids,
        new_tracks,
        removed_tracks_ids
    ):
        prediction = None
        mask_avg_prob_dict = {}
        prediction_colors_preserved = None

        if self.tracklet_mask_dict is None:
            self.tracklet_mask_dict = {}

        raw_frame = img_info['raw_img']
        raw_frame_prev = img_info_prev['raw_img']

        frame_cutie = self._resize_frame_for_cutie(raw_frame)
        frame_cutie_prev = self._resize_frame_for_cutie(raw_frame_prev)

        frame_torch = image_to_torch(frame_cutie, device=self.device)
        frame_torch_prev = image_to_torch(frame_cutie_prev, device=self.device)

        if (
            frame_id == self.SAM_START_FRAME + 1 + self.init_delay_counter
            and online_tlwhs is not None
        ):
            prediction = self.initialize_first_masks(
                frame_torch,
                frame_torch_prev,
                img_info_prev,
                online_tlwhs,
                online_ids
            )

        elif frame_id > self.SAM_START_FRAME + 1 + self.init_delay_counter:
            self.add_new_masks(
                frame_torch_prev,
                img_info_prev,
                online_tlwhs,
                online_ids,
                new_tracks
            )

            self.remove_masks(removed_tracks_ids)

            with torch.no_grad():
                prediction = self.processor.step(frame_torch.clone())

        if prediction is not None:
            (
                prediction,
                mask_avg_prob_dict,
                prediction_colors_preserved
            ) = self.post_process_mask(prediction)

        return (
            prediction,
            self.tracklet_mask_dict.copy(),
            mask_avg_prob_dict,
            prediction_colors_preserved
        )

    # ============================================================
    # INITIALIZE FIRST MASKS
    # ============================================================

    def initialize_first_masks(
        self,
        frame_torch,
        frame_torch_prev,
        img_info_prev,
        online_tlwhs,
        online_ids
    ):
        raw_frame = img_info_prev['raw_img']

        self.sam_predictor.set_image(raw_frame)

        image_boxes_list = []
        new_tracks_id = []

        for i, ot in enumerate(online_tlwhs):
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(ot, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(ot, track_BBs_with_lower_bottom)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue

            image_boxes_list.append([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]])
            new_tracks_id.append(online_ids[i])

        if len(image_boxes_list) == 0:
            self.init_delay_counter += 1
            return None

        image_boxes = torch.tensor(
            image_boxes_list, device=self.sam.device, dtype=torch.float32
        )

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            image_boxes, raw_frame.shape[:2]
        )

        with torch.no_grad():
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

        # --------------------------------------------------------
        # Assign a fresh, permanent stable_id per detected mask,
        # IN ORDER -- this order fixes the channel/position order
        # for both the one-hot input and current_object_list_cutie.
        # --------------------------------------------------------
        stable_ids = [self._new_stable_id() for _ in range(len(masks))]

        mask = np.zeros(masks[0].shape, dtype=np.int32)

        for mi in range(len(masks)):
            current_mask = masks[mi].detach().cpu().numpy().astype(np.int32)
            current_mask[current_mask > 0] = mi + 1  # position-based paint, 1..N
            non_occupied = (mask == 0).astype(np.int32)
            mask += current_mask * non_occupied

        mask = mask.squeeze(0)

        self.num_objects = len(masks)
        # position i (0-indexed) -> stable_id at that position
        self.current_object_list_cutie = list(stable_ids)

        self.tracklet_mask_dict = dict(zip(new_tracks_id, stable_ids))

        mask_cutie = self._resize_label_mask_for_cutie(mask)

        mask_torch = index_numpy_to_one_hot_torch(
            mask_cutie, self.num_objects + 1
        ).to(self.device).clone()

        with torch.no_grad():
            _ = self.processor.step(
                frame_torch_prev.clone(),
                mask_torch[1:].clone(),
                idx_mask=False
            )
            prediction = self.processor.step(frame_torch.clone())

        return prediction

    # ============================================================
    # ADD NEW MASKS
    # ============================================================

    def add_new_masks(
        self,
        frame_torch_prev,
        img_info_prev,
        online_tlwhs,
        online_ids,
        new_tracks
    ):
        if self.tracklet_mask_dict is None:
            self.tracklet_mask_dict = {}

        if online_tlwhs is None or online_ids is None:
            return

        if len(new_tracks) == 0 and len(self.awaiting_mask_tracklet_ids) == 0:
            return

        raw_frame = img_info_prev['raw_img']
        self.sam_predictor.set_image(raw_frame)

        image_boxes_list = []
        new_tracks_id = []

        # --------------------------------------------------------
        # Awaiting masks
        # --------------------------------------------------------
        for amti in self.awaiting_mask_tracklet_ids:
            if amti not in online_ids:
                continue

            amt_index = online_ids.index(amti)
            amt_tlwh = online_tlwhs[amt_index]

            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(amt_tlwh, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(amt_tlwh, track_BBs_with_lower_bottom)

            if overlap < MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                image_boxes_list.append(
                    [amt_tlwh[0], amt_tlwh[1], amt_tlwh[0] + amt_tlwh[2], amt_tlwh[1] + amt_tlwh[3]]
                )
                new_tracks_id.append(amti)

        for nti in new_tracks_id:
            if nti in self.awaiting_mask_tracklet_ids:
                self.awaiting_mask_tracklet_ids.remove(nti)

        # --------------------------------------------------------
        # Newly created tracks
        # --------------------------------------------------------
        for nt in new_tracks:
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(nt.last_det_tlwh, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(nt.last_det_tlwh, track_BBs_with_lower_bottom)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                if nt.track_id not in self.awaiting_mask_tracklet_ids:
                    self.awaiting_mask_tracklet_ids.append(nt.track_id)
                continue

            image_boxes_list.append(
                [
                    nt.last_det_tlwh[0],
                    nt.last_det_tlwh[1],
                    nt.last_det_tlwh[0] + nt.last_det_tlwh[2],
                    nt.last_det_tlwh[1] + nt.last_det_tlwh[3]
                ]
            )
            new_tracks_id.append(nt.track_id)

        if len(image_boxes_list) == 0:
            return

        image_boxes = torch.tensor(
            image_boxes_list, device=self.sam.device, dtype=torch.float32
        )

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            image_boxes, raw_frame.shape[:2]
        )

        with torch.no_grad():
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

        # --------------------------------------------------------
        # ONE stable_id per new mask. This id is used everywhere:
        # painted pixel value, tracklet_mask_dict value, and the
        # entry appended to current_object_list_cutie (in the same
        # order the one-hot channels will be constructed).
        # --------------------------------------------------------
        new_stable_ids = [self._new_stable_id() for _ in range(len(masks))]

        mask_extra = np.zeros(masks[0].shape, dtype=np.int32)

        for mi in range(len(masks)):
            current_mask = masks[mi].detach().cpu().numpy().astype(np.int32)
            current_mask[current_mask > 0] = new_stable_ids[mi]
            non_occupied = (mask_extra == 0).astype(np.int32)
            mask_extra += current_mask * non_occupied

        mask_extra = mask_extra.squeeze(0)

        mask_extra_cutie = self._resize_label_mask_for_cutie(mask_extra)

        if self.mask_prediction_prev_frame_cutie is None:
            return

        self.mask_prediction_prev_frame_cutie[mask_extra_cutie > 0] = (
            mask_extra_cutie[mask_extra_cutie > 0]
        )

        # --------------------------------------------------------
        # current_object_list_cutie order MUST match the one-hot
        # channel order below. index_numpy_to_one_hot_torch builds
        # channels for label values 1..num_classes-1 in ascending
        # order of the label value itself, so we build the one-hot
        # target size from stable_id space directly and keep
        # current_object_list_cutie as "all currently-alive stable
        # ids, ascending" to stay consistent with that.
        # --------------------------------------------------------
        self.current_object_list_cutie = sorted(
            set(self.current_object_list_cutie) | set(new_stable_ids)
        )
        self.num_objects = len(self.current_object_list_cutie)

        max_stable_id = max(self.current_object_list_cutie)

        mask_prev_extended_torch = index_numpy_to_one_hot_torch(
            self.mask_prediction_prev_frame_cutie,
            max_stable_id + 1
        ).to(self.device).clone()

        # Only keep the channels for stable ids we actually track,
        # in ascending order, matching current_object_list_cutie.
        keep_channels = [0] + self.current_object_list_cutie
        mask_prev_extended_torch = mask_prev_extended_torch[keep_channels]

        with torch.no_grad():
            _ = self.processor.step(
                frame_torch_prev.clone(),
                mask_prev_extended_torch[1:].clone(),
                objects=self.current_object_list_cutie,
                idx_mask=False
            )

        for track_id, stable_id in zip(new_tracks_id, new_stable_ids):
            self.tracklet_mask_dict[track_id] = stable_id

    # ============================================================
    # REMOVE MASKS
    # ============================================================

    def remove_masks(self, removed_tracks_ids):
        if self.tracklet_mask_dict is None:
            self.tracklet_mask_dict = {}

        if len(removed_tracks_ids) == 0:
            return

        stable_ids_to_remove = [
            self.tracklet_mask_dict[i]
            for i in self.tracklet_mask_dict.keys()
            if i in removed_tracks_ids
        ]

        if len(stable_ids_to_remove) == 0:
            return

        purge_activated, tmp_keep_idx, obj_keep_idx = (
            self.processor.object_manager.purge_selected_objects(stable_ids_to_remove)
        )

        if purge_activated:
            self.processor.memory.purge_except(obj_keep_idx)

        # --------------------------------------------------------
        # Do NOT trust obj_keep_idx's ordering/values as a new
        # numbering scheme. Just drop the removed ids from OUR own
        # authoritative list -- stable ids are never reused, so
        # this can never collide with a live object.
        # --------------------------------------------------------
        self.current_object_list_cutie = [
            sid for sid in self.current_object_list_cutie
            if sid not in stable_ids_to_remove
        ]
        self.num_objects = len(self.current_object_list_cutie)

        # Only delete the entries -- NEVER renumber/decrement the
        # remaining stable ids. Renumbering is what caused new
        # masks to collide with still-alive objects after removal.
        for track_id in list(self.tracklet_mask_dict.keys()):
            if self.tracklet_mask_dict[track_id] in stable_ids_to_remove:
                del self.tracklet_mask_dict[track_id]

    # ============================================================
    # POST PROCESS
    # ============================================================

    def post_process_mask(self, prediction):
        self.prediction = prediction

        mask_avg_prob_dict = self.get_mask_avg_prob(prediction)

        prediction_cutie_mask = torch_prob_to_numpy_mask(prediction)
        self.mask_prediction_prev_frame_cutie = prediction_cutie_mask.copy()

        prediction_original = self._resize_prediction_to_original(prediction)
        self.mask_prediction_prev_frame = prediction_original.copy()

        # Remap Cutie's POSITION-indexed output to stable_ids so it
        # lines up with tracklet_mask_dict.
        prediction_colors_preserved = self.adjust_mask_colors(prediction_original)

        return prediction_original, mask_avg_prob_dict, prediction_colors_preserved

    # ============================================================
    # MASK SCORE
    # ============================================================

    def get_mask_avg_prob(self, prediction):
        mask_avg_prob_dict = {}

        if self.tracklet_mask_dict is None:
            return mask_avg_prob_dict

        mask_maxes = torch.max(prediction, dim=0).indices
        stable_to_position = self._stable_to_position_map()

        for stable_id in self.tracklet_mask_dict.values():
            position = stable_to_position.get(stable_id)

            if position is None or position >= prediction.shape[0]:
                continue

            selected = prediction[position][mask_maxes == position]

            if selected.numel() == 0:
                continue

            average_mask_v_score = selected.mean().item()

            if not np.isnan(average_mask_v_score):
                mask_avg_prob_dict[stable_id] = average_mask_v_score

        return mask_avg_prob_dict

    # ============================================================
    # COLORS / IDENTITY REMAP
    # ============================================================

    def adjust_mask_colors(self, prediction):
        """
        `prediction` (from torch_prob_to_numpy_mask) is labeled by
        POSITION (1..num_objects). Remap every pixel to the stable_id
        that tracklet_mask_dict actually uses, via the single fresh
        position->stable_id map. This is the only remap step, and
        it can never drift because it's derived every call from
        current_object_list_cutie, not accumulated incrementally.
        """
        position_to_stable = self._position_to_stable_map()

        out = np.zeros_like(prediction)

        # process descending so a stable_id that happens to equal
        # an original position value can't be double-remapped
        for position in sorted(position_to_stable.keys(), reverse=True):
            out[prediction == position] = position_to_stable[position]

        return out


# ================================================================
# OVERLAP HELPERS (unchanged)
# ================================================================

def get_tracklets_with_lower_bottom(new_tracklet_tlwh, online_tlwhs):
    nt_y = new_tracklet_tlwh[1]
    nt_h = new_tracklet_tlwh[3]

    track_BBs_with_lower_bottom = []
    nt_bottom = nt_y + nt_h

    for ot in online_tlwhs:
        if ot[1] + ot[3] > nt_bottom:
            track_BBs_with_lower_bottom.append(ot)

    return track_BBs_with_lower_bottom


def get_overlap_with_lower_bottom_tracklets(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    overlap = 0

    if OVERLAP_MEASURE_VARIANT == 1:
        overlap = get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom)
    elif OVERLAP_MEASURE_VARIANT == 2:
        overlap = get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom)

    return overlap


def get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = new_tracklet_tlwh[0]
    nt_y = new_tracklet_tlwh[1]
    nt_w = new_tracklet_tlwh[2]
    nt_h = new_tracklet_tlwh[3]

    max_overlap_part = 0

    for lb in track_BBs_with_lower_bottom:
        x_dist = min(nt_x + nt_w, lb[0] + lb[2]) - max(nt_x, lb[0])
        y_dist = min(nt_y + nt_h, lb[1] + lb[3]) - max(nt_y, lb[1])

        if x_dist < 0 or y_dist < 0:
            overlap_area = 0
        else:
            overlap_area = x_dist * y_dist

        denom = max(1, nt_w * nt_h)
        overlap_part = overlap_area / denom

        if max_overlap_part < overlap_part:
            max_overlap_part = overlap_part
            if max_overlap_part == 1:
                break

    return max_overlap_part


def get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = int(new_tracklet_tlwh[0])
    nt_y = int(new_tracklet_tlwh[1])
    nt_w = int(new_tracklet_tlwh[2])
    nt_h = int(new_tracklet_tlwh[3])

    point_overlap_counter = 0

    rows = range(nt_y, nt_y + nt_h, OVERLAP_VARIANT_2_GRID_STEP)
    cols = range(nt_x, nt_x + nt_w, OVERLAP_VARIANT_2_GRID_STEP)

    total_points = len(rows) * len(cols)

    if total_points == 0:
        return 0

    for grid_row in rows:
        for grid_col in cols:
            for lb in track_BBs_with_lower_bottom:
                if lb[0] <= grid_col <= lb[0] + lb[2] and lb[1] <= grid_row <= lb[1] + lb[3]:
                    point_overlap_counter += 1
                    break

    return point_overlap_counter / total_points