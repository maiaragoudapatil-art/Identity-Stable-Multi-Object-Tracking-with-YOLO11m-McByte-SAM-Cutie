import numpy as np
np.set_printoptions(threshold=np.inf)

from collections import deque
import os.path as osp
import copy
import cv2

from .kalman_filter import KalmanFilter
from yolox.tracker import matching as matching
from yolox.tracker.gmc import GMC
from .basetrack import BaseTrack, TrackState

### Constants ###
MIN_MASK_AVG_CONF = 0.6
MIN_MM1 = 0.9
MIN_MM2 = 0.05

MAX_COST_1ST_ASSOC_STEP = 0.98
MAX_COST_2ND_ASSOC_STEP = 0.5
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.7

# ============================================================
# ID-STABILITY SETTINGS
# ============================================================
# Appearance is used as a real identity gate, not only when the
# IoU matrix happens to be ambiguous.
#
# NOTE on tuning: appearance here is an HSV clothing-color
# histogram. It CANNOT distinguish two people wearing similar
# colors (e.g. same team kit, same dark jacket). If your footage
# has visually similar people, lean more on APP_HARD_GATE being
# strict (reject only clear mismatches) and less on using
# appearance to break close ties -- ties between similarly
# dressed people should be resolved by motion/IoU, not color.
APP_WEIGHT = 0.60
IOU_WEIGHT = 0.40
APP_HARD_GATE = 0.45

# Maximum normalized center displacement allowed for an identity match.
# This prevents a predicted track from jumping onto a nearby person during occlusion.
MAX_CENTER_JUMP = 2.25
RECOVERY_MAX_CENTER_JUMP = MAX_CENTER_JUMP

# Do not contaminate an identity template with a clearly different crop.
APPEARANCE_UPDATE_GATE = 0.38

# Recovery (post-occlusion re-identification) is inherently riskier
# than frame-to-frame matching: more time has passed, appearance
# memory is older, and the person may be standing right next to
# someone else who looks similar. The recovery gate must therefore
# be AT LEAST as strict as the normal gate, never looser.
RECOVERY_APP_GATE = 0.50
RECOVERY_MAX_COST = 0.55
RECOVERY_MARGIN = 0.08
# Absolute-confidence bypass for the ambiguity margin. Keep this
# tight -- it exists only for genuinely unambiguous matches, not as
# a general escape hatch (a looser value here defeats the margin
# check entirely whenever two similar-looking people are both lost).
RECOVERY_ABS_CONFIDENT = 0.30
APPEARANCE_MEMORY = 20

# Appearance descriptor layout (must match _appearance_descriptor).
_HSV_REGION_BINS = 18 * 8   # 144
_NUM_HSV_REGIONS = 3
_GRAY_BINS = 16
_DESCRIPTOR_LEN = _NUM_HSV_REGIONS * _HSV_REGION_BINS + _GRAY_BINS  # 448


def _best_detection_mask(prediction_mask, tlbr):
    """Find the Cutie/SAM instance label with the largest overlap in a box."""
    if prediction_mask is None:
        return None
    try:
        pm = np.asarray(prediction_mask)
        if pm.ndim != 2:
            return None
        h, w = pm.shape
        x1, y1, x2, y2 = [int(v) for v in tlbr]
        x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
        roi = pm[y1:y2, x1:x2]
        vals = roi[roi > 0]
        if vals.size == 0:
            return None
        labels, counts = np.unique(vals, return_counts=True)
        label = int(labels[np.argmax(counts)])
        if int(np.max(counts)) < 50:
            return None
        return (pm == label).astype(np.uint8)
    except Exception:
        return None


def _appearance_descriptor(frame, tlbr, person_mask=None):
    """Mask-aware player appearance descriptor.

    Layout (fixed, must match _appearance_distance):
        [region0 HSV hist (144)] [region1 HSV hist (144)]
        [region2 HSV hist (144)] [0.35 * gray hist (16)]
    Total length: 448.
    """
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in tlbr]
        x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 2, min(w, x2)); y2 = max(y1 + 2, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]

        if person_mask is not None:
            pm = np.asarray(person_mask)
            if pm.ndim == 2 and pm.shape[0] >= y2 and pm.shape[1] >= x2:
                local_mask = (pm[y1:y2, x1:x2] > 0).astype(np.uint8)
            else:
                local_mask = np.ones((ch, cw), dtype=np.uint8)
        else:
            local_mask = np.ones((ch, cw), dtype=np.uint8)

        bx = max(1, int(cw * 0.06)); by = max(1, int(ch * 0.05))
        border = np.zeros((ch, cw), dtype=np.uint8)
        border[by:max(by + 1, ch - by), bx:max(bx + 1, cw - bx)] = 1
        local_mask *= border
        if int(local_mask.sum()) < 50:
            local_mask = border

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        features = []
        for frac0, frac1 in ((0.08, 0.48), (0.30, 0.72), (0.55, 0.95)):
            a = max(0, int(ch * frac0)); b = min(ch, max(a + 1, int(ch * frac1)))
            hist = cv2.calcHist([hsv[a:b]], [0, 1], local_mask[a:b],
                                 [18, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, None).flatten().astype(np.float32)
            features.append(hist)
        gh = cv2.calcHist([gray], [0], local_mask, [16], [0, 256])
        gh = cv2.normalize(gh, None).flatten().astype(np.float32)
        features.append(0.35 * gh)
        desc = np.concatenate(features).astype(np.float32)
        norm = np.linalg.norm(desc)
        if norm > 1e-8:
            desc /= norm
        return desc
    except Exception:
        return None


def _appearance_distance(a, b):
    """Bhattacharyya distance per semantic region (3 body-region HSV
    histograms + 1 grayscale histogram), combined with fixed weights.

    Region boundaries here MUST match _appearance_descriptor's layout.
    Previously this function assumed a fictitious trailing 6-value
    "statistics" feature that never existed in the descriptor, and
    split the remaining histogram in half without regard to the
    actual region boundaries -- comparing mismatched fragments of
    different body regions against each other. That silently
    degraded the appearance signal the whole identity gate depends
    on. Fixed to compare each real region against its counterpart.
    """
    if a is None or b is None:
        return None
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            return None

        if a.shape[0] != _DESCRIPTOR_LEN:
            # Unknown/legacy layout -- fall back to a generic
            # Bhattacharyya-style coefficient over the full vector
            # rather than guessing at a split point.
            bc = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
            return float(max(0.0, min(1.0, 1.0 - bc)))

        region_dists = []
        for r in range(_NUM_HSV_REGIONS):
            start = r * _HSV_REGION_BINS
            end = start + _HSV_REGION_BINS
            d = cv2.compareHist(a[start:end], b[start:end], cv2.HISTCMP_BHATTACHARYYA)
            region_dists.append(float(d))

        gray_start = _NUM_HSV_REGIONS * _HSV_REGION_BINS
        d_gray = cv2.compareHist(a[gray_start:], b[gray_start:], cv2.HISTCMP_BHATTACHARYYA)

        # Torso region weighted a bit more heavily -- largest, most
        # stable clothing area, least affected by limb motion/pose.
        d_hist = (
            0.30 * region_dists[0]
            + 0.40 * region_dists[1]
            + 0.30 * region_dists[2]
        )

        return float(0.85 * d_hist + 0.15 * d_gray)
    except Exception:
        return None


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.last_det_tlwh = tlwh ## Extra added
        self.appearance = None
        self.appearance_gallery = deque(maxlen=APPEARANCE_MEMORY)

    def update_appearance(self, descriptor):
        """Update identity memory without replacing it with one noisy crop."""
        if descriptor is None:
            return
        try:
            desc = np.asarray(descriptor, dtype=np.float32).reshape(-1)
            if desc.size == 0:
                return
            # Never let a wrong-person crop poison the identity memory.
            # The first descriptor is always accepted; later descriptors must
            # remain reasonably close to the established identity.
            if self.appearance is not None:
                d = _appearance_distance(self.appearance, desc)
                if d is not None and d > APPEARANCE_UPDATE_GATE:
                    return
            self.appearance_gallery.append(desc.copy())
            # Stable identity template: median over recent observations.
            self.appearance = np.median(
                np.stack(list(self.appearance_gallery), axis=0), axis=0
            ).astype(np.float32)
        except Exception:
            pass

    def appearance_distance(self, descriptor):
        if descriptor is None:
            return None
        candidates = list(getattr(self, 'appearance_gallery', []))
        if getattr(self, 'appearance', None) is not None:
            candidates.append(self.appearance)
        if not candidates:
            return None
        vals = []
        for old in candidates:
            d = _appearance_distance(old, descriptor)
            if d is not None:
                vals.append(d)
        return min(vals) if vals else None

    def center_distance_normalized(self, descriptor_tlwh):
        try:
            a = self.xywh
            b = np.asarray(descriptor_tlwh, dtype=np.float32).copy()
            b[:2] += b[2:] / 2.0
            scale = max(20.0, float(max(self.tlwh[2], self.tlwh[3], b[2], b[3])))
            return float(np.linalg.norm(a[:2] - b[:2]) / scale)
        except Exception:
            return float("inf")

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0 ## Extra added from botsort
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0 ## Extra added from botsort
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov


    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # new_id=False is essential for occlusion recovery: keep the old ID.
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh ## Extra added
        self.update_appearance(getattr(new_track, 'appearance', None))

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.last_det_tlwh = new_track.tlwh ## Extra added
        self.update_appearance(getattr(new_track, 'appearance', None))

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class McByteLogger(object):
    def __init__(self, log_file_path):
        self.file = open(log_file_path, "w")
        np.set_printoptions(linewidth=1000)

    def __del__(self):
        self.file.close()

    def log_info(self):
        pass

    def log_frame_no(self, frame_no):
        self.file.write("\n\n= = = = = = Frame number: " + str(frame_no) + " = = = = =\n\n")

    def log_mask_info(self, tracklet_mask_dict):
        self.file.write("tracklet_id -> mask_number:\n")
        if tracklet_mask_dict is None:
            self.file.write("< tracklet_mask_dict is None, probably frame(s) before creating the masks with SAM >")
        else:
            for k, v in tracklet_mask_dict.items():
                self.file.write(str(k) + " -> " + str(v) + ", ")
        self.file.write("\n\n")

    def log_dists(self, dists, mask_match_included, which_association, frame_no):
        self.file.write("Association step: " + str(which_association) + " (frame " + str(frame_no) + ")" + "\nMask match included: " + str(mask_match_included) + "\n\n")
        self.file.write(str(dists) + "\n\n")

    def log_matches(self, matches, u_track, u_detection, strack_pool_ids):
        self.file.write("matrix row -> tracklet_id:\n")
        for i in range(len(strack_pool_ids)):
            self.file.write(str(i) + " -> " + str(strack_pool_ids[i]) + ", ")
        self.file.write("\n\n")

        self.file.write("matches [row column] [track det]:\n")
        for match in matches:
            self.file.write(str(match) + "\n")
        self.file.write("u_track:\n" + str(u_track) + "\n")
        self.file.write("u_detection:\n" + str(u_detection) + "\n")

    def log_det_conf_scores(self, detections):
        self.file.write("Detection confidence scores:\n")
        for i, det in enumerate(detections):
            self.file.write(str(i) + " : " + str(np.round(det.score, decimals=2)) + "\t")
        self.file.write("\n\n")
        self.file.write("- - - - - - - - - -\n\n")

    def log_local_update_trackets_ids(self, activated, refind, lost, removed):
        self.file.write("activated_stracks: ")
        for track in activated:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nrefind_stracks: ")
        for track in refind:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nlost_stracks: ")
        for track in lost:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nremoved_stracks: ")
        for track in removed:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\n\n")

    def log_state_tracklets_ids(self, tracked, lost, removed):
        self.file.write("self.track_stracks: ")
        for track in tracked:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nself.lost_stracks: ")
        for track in lost:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\nself.removed_stracks: ")
        for track in removed:
            self.file.write(str(track.track_id) + ", ")
        self.file.write(".\n\n")


class McByteTracker(object):
    def __init__(self, args, save_folder, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # Do not add the historical +0.1 offset here. The supplied
        # sports checkpoint can produce low-confidence person scores; adding
        # 0.1 silently prevents those detections from ever becoming tracks.
        self.det_thresh = max(float(args.track_thresh), 0.001)
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # For camera motion compensation
        self.gmc = GMC(method=args.cmc_method, verbose=None)

        self.save_folder = save_folder
        log_file_path = osp.join(self.save_folder, "logging_info.txt")
        self.logger = McByteLogger(log_file_path)


    def conditioned_assignment(self, dists, max_cost, strack_pool, detections,
                               prediction_mask, tracklet_mask_dict,
                               mask_avg_prob_dict, img_info, frame_img=None):
        """Association with identity-aware gating.

        Appearance is checked for every spatially-plausible candidate, not
        only when the IoU matrix happens to be ambiguous -- a person's
        predicted box can overlap the *wrong* person during occlusion, and
        that's exactly when a pure-IoU match would lock in the wrong ID.

        IMPORTANT: the "which track most strongly matches this detection"
        veto below is restricted to tracks that were ALSO within max_cost of
        this detection (i.e. genuine spatial competitors). Comparing against
        the entire pool -- including tracks nowhere near this detection --
        let a distant, similarly-dressed person (same jersey/jacket color)
        veto the correct, spatially-obvious match. That was a real source
        of incorrect rejections/ID loss, not a safety net.
        """
        dists_cp = np.asarray(dists, dtype=np.float32).copy()
        tracklet_mask_dict = tracklet_mask_dict if isinstance(tracklet_mask_dict, dict) else {}
        mask_avg_prob_dict = mask_avg_prob_dict if isinstance(mask_avg_prob_dict, dict) else {}

        if dists_cp.size == 0:
            return matching.linear_assignment(dists_cp, thresh=max_cost) + (dists_cp,)

        # Computed once per call, not per (i, j) pair.
        mask_values = None
        if prediction_mask is not None:
            try:
                mask_values = np.unique(prediction_mask)
            except Exception:
                mask_values = None

        for i in range(dists_cp.shape[0]):
            for j in range(dists_cp.shape[1]):
                base = float(dists[i, j])
                if base > max_cost:
                    continue

                strack = strack_pool[i]
                det = detections[j]
                det_app = getattr(det, 'appearance', None)
                app_dist = None
                if hasattr(strack, 'appearance_distance'):
                    app_dist = strack.appearance_distance(det_app)
                else:
                    app_dist = _appearance_distance(
                        getattr(strack, 'appearance', None), det_app
                    )

                # Hard identity gate. A visually different person must not
                # inherit this track merely because the Kalman/IoU position
                # happened to overlap during an occlusion.
                if app_dist is not None and app_dist > APP_HARD_GATE:
                    dists_cp[i, j] = 1.5
                    continue

                # Identity cannot jump an implausible distance in one frame.
                # This is the important protection against ID shifting when
                # two people cross/occlude each other.
                try:
                    center_jump = strack.center_distance_normalized(det.tlwh)
                except Exception:
                    center_jump = float("inf")
                if center_jump > MAX_CENTER_JUMP:
                    dists_cp[i, j] = 1.5
                    continue

                # Always combine motion/IoU and appearance. Never bypass
                # appearance because a pair happens to be unique.
                if app_dist is not None:
                    dists_cp[i, j] = IOU_WEIGHT * base + APP_WEIGHT * app_dist

                    # If several track IDs could claim the same detection,
                    # prefer the identity with the strongest appearance
                    # match -- but only among tracks that were themselves
                    # spatially plausible candidates for this detection.
                    other_apps = []
                    for oi in range(dists_cp.shape[0]):
                        if oi == i:
                            continue
                        if float(dists[oi, j]) > max_cost:
                            # Not a real competitor for this detection --
                            # don't let it veto a genuine spatial match.
                            continue
                        other = strack_pool[oi]
                        od = other.appearance_distance(det_app) if hasattr(other, 'appearance_distance') else None
                        if od is not None:
                            other_apps.append(float(od))
                    if other_apps and app_dist > min(other_apps) + 0.12:
                        dists_cp[i, j] = 1.5
                        continue

                # Existing McByte mask cue.
                strack_id = strack.track_id
                if (prediction_mask is not None and
                    strack_id in tracklet_mask_dict and
                    mask_values is not None):
                    strack_mask_id = tracklet_mask_dict[strack_id]
                    if strack_mask_id in mask_values:
                        if mask_avg_prob_dict.get(strack_mask_id, 0.0) >= MIN_MASK_AVG_CONF:
                            img_h, img_w = img_info[0], img_info[1]
                            x, y, w, h = det.tlwh
                            x = max(0, int(x))
                            y = max(0, int(y))
                            hor_bound = min(img_w, x + max(1, int(w)))
                            ver_bound = min(img_h, y + max(1, int(h)))
                            if hor_bound > x and ver_bound > y:
                                total_mask = float((prediction_mask == strack_mask_id).sum())
                                box_area = float((ver_bound-y) * (hor_bound-x))
                                if total_mask > 0 and box_area > 0:
                                    inside = float((prediction_mask[y:ver_bound, x:hor_bound] == strack_mask_id).sum())
                                    mask_match_opt_1 = inside / total_mask
                                    mask_match_opt_2 = inside / box_area
                                    if mask_match_opt_2 >= MIN_MM2:
                                        if mask_match_opt_1 < MIN_MM1:
                                            dists_cp[i, j] = 1.5
                                            continue
                                        dists_cp[i, j] -= min(mask_match_opt_2, 0.25)

        matches, u_track, u_detection = matching.linear_assignment(
            dists_cp, thresh=max_cost
        )
        return matches, u_track, u_detection, dists_cp


    def update(self, output_results, img_info, img_size, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, frame_img, vis_type, dets_from_file=False):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        assoc1_dets = []
        assoc2_dets = []
        assoc3_dets = []
        init_track_dets_acc = []
        init_track_dets_rej = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]

        if not dets_from_file:
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        self.logger.log_frame_no(self.frame_id)
        self.logger.log_state_tracklets_ids(self.tracked_stracks, self.lost_stracks, self.removed_stracks)
        self.logger.log_mask_info(tracklet_mask_dict)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # Extract appearance once per high-confidence detection.
        if frame_img is not None:
            for det in detections:
                det_mask = _best_detection_mask(prediction_mask, det.tlbr)
                det.appearance = _appearance_descriptor(frame_img, det.tlbr, det_mask)

        ''' Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        ''' Step 2: First association, with high score detection boxes'''
        # IMPORTANT: lost tracks are NOT allowed to participate in normal
        # IoU association. They are recovered separately using identity
        # memory below. This prevents a lost ID from hijacking another person.
        strack_pool = list(tracked_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        try:
            warp = self.gmc.apply(frame_img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        except Exception:
            pass

        # Do visualize all considered tracklets before KF correction (update):
        if vis_type == 'full':
            strack_pool_before_correction = copy.deepcopy(strack_pool)
            unconfirmed_before_correction = copy.deepcopy(unconfirmed)
            all_considered_tracklets_before_correction = joint_stracks(strack_pool_before_correction, unconfirmed_before_correction)
        else:
            all_considered_tracklets_before_correction = None

        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)

        if vis_type == 'full':
            assoc1_dets = [det for det in detections]  # For detection visualization (1/5)
        self.logger.log_dists(dists, mask_match_included=False, which_association=1, frame_no=self.frame_id)

        matches, u_track, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_1ST_ASSOC_STEP, strack_pool, detections, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info, frame_img)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=1, frame_no=self.frame_id)

        strack_pool_ids = [s.track_id for s in strack_pool]
        self.logger.log_matches(matches, u_track, u_detection, strack_pool_ids)
        self.logger.log_det_conf_scores(detections)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        if frame_img is not None:
            for det in detections_second:
                det_mask = _best_detection_mask(prediction_mask, det.tlbr)
                det.appearance = _appearance_descriptor(frame_img, det.tlbr, det_mask)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        if vis_type == 'full':
            assoc2_dets = [det for det in detections_second] # For detection visualization (2/5)
        self.logger.log_dists(dists, mask_match_included=False, which_association=2, frame_no=self.frame_id)

        matches, u_track, u_detection_second, dists_cp = self.conditioned_assignment(dists, MAX_COST_2ND_ASSOC_STEP, r_tracked_stracks, detections_second, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info, frame_img)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=2, frame_no=self.frame_id)

        strack_pool_ids = [s.track_id for s in r_tracked_stracks]
        self.logger.log_matches(matches, u_track, u_detection_second, strack_pool_ids)
        self.logger.log_det_conf_scores(detections_second)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)

        # NOTE: At this stage, the unconfirmed tracklets do not have their own masks yet

        if vis_type == 'full':
            assoc3_dets = [det for det in detections]  # For detection visualization (3/5)
        self.logger.log_dists(dists, mask_match_included=False, which_association=3, frame_no=self.frame_id)

        matches, u_unconfirmed, u_detection, dists_cp = self.conditioned_assignment(dists, MAX_COST_UNCONFIRMED_ASSOC_STEP, unconfirmed, detections, prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info, frame_img)
        self.logger.log_dists(dists_cp, mask_match_included=True, which_association=3, frame_no=self.frame_id)

        strack_pool_ids = [s.track_id for s in unconfirmed]
        self.logger.log_matches(matches, u_unconfirmed, u_detection, strack_pool_ids)
        self.logger.log_det_conf_scores(detections)

        new_confirmed_tracks = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            new_confirmed_tracks.append(unconfirmed[itracked])
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        """ Step 5: Recover lost tracks BEFORE creating new IDs """

        self.feature_db_new_inits = {}

        # ============================================================
        # HARD LOST-ID RESERVATION / RECOVERY
        # ============================================================
        # A lost ID must get the FIRST chance to reclaim a detection.
        # Otherwise ByteTrack can create a new ID for the same person and
        # the old ID can reappear later, producing exactly this pattern:
        #
        #     ID 10 -> missing -> ID 13 -> ID 10
        #
        # For short gaps, motion/position is more reliable than a color
        # histogram. Therefore a spatially plausible lost track is allowed
        # to reclaim the detection even when appearance is noisy.
        # New IDs are created only after this reservation step.
        # ============================================================

        unmatched_detections = list(u_detection)

        recovery_pool = []
        seen_recovery_ids = set()
        for t in list(self.lost_stracks) + list(lost_stracks):
            if t.state != TrackState.Lost:
                continue
            tid = int(t.track_id)
            if tid in seen_recovery_ids:
                continue
            seen_recovery_ids.add(tid)
            recovery_pool.append(t)

        if unmatched_detections and recovery_pool:
            # Predict old lost tracks only once. Tracks placed in local
            # lost_stracks this frame were already predicted above.
            previous_lost = [
                t for t in recovery_pool
                if int(getattr(t, "frame_id", -1)) < int(self.frame_id)
            ]
            if previous_lost:
                try:
                    STrack.multi_predict(previous_lost)
                    try:
                        warp = self.gmc.apply(frame_img, dets)
                        STrack.multi_gmc(previous_lost, warp)
                    except Exception:
                        pass
                except Exception:
                    pass

            candidates = []
            for di in unmatched_detections:
                det = detections[di]
                det_app = getattr(det, "appearance", None)
                for lost_track in recovery_pool:
                    gap = max(0, int(self.frame_id - getattr(lost_track, "end_frame", self.frame_id)))
                    center_jump = lost_track.center_distance_normalized(det.tlwh)

                    try:
                        iou_d = float(matching.iou_distance([lost_track], [det])[0, 0])
                        pred_iou = max(0.0, min(1.0, 1.0 - iou_d))
                    except Exception:
                        pred_iou = 0.0

                    app_dist = lost_track.appearance_distance(det_app) if det_app is not None else None

                    # ----------------------------------------------------
                    # SHORT-GAP HARD RESERVATION
                    # ----------------------------------------------------
                    # If the detection is still spatially close to the old
                    # track, reserve that old ID. This is the key protection
                    # against duplicate IDs during a 1-5 frame miss.
                    short_gap = gap <= 5
                    spatial_match = (
                        center_jump <= 1.65 or
                        pred_iou >= 0.05
                    )

                    if short_gap and spatial_match:
                        if app_dist is None:
                            cost = 0.45 * (1.0 - pred_iou) + 0.55 * min(center_jump / 1.65, 1.0)
                        else:
                            # Appearance helps rank candidates but cannot
                            # override strong short-gap spatial continuity.
                            cost = (
                                0.20 * float(app_dist)
                                + 0.45 * (1.0 - pred_iou)
                                + 0.35 * min(center_jump / 1.65, 1.0)
                            )
                        candidates.append((
                            cost, di, lost_track, center_jump,
                            pred_iou, app_dist, gap, True
                        ))
                        continue

                    # ----------------------------------------------------
                    # LONGER-GAP RE-ID
                    # ----------------------------------------------------
                    if center_jump > RECOVERY_MAX_CENTER_JUMP:
                        continue

                    if app_dist is not None:
                        if app_dist > RECOVERY_APP_GATE and pred_iou < 0.20:
                            continue
                        cost = (
                            0.55 * float(app_dist)
                            + 0.30 * (1.0 - pred_iou)
                            + 0.15 * min(center_jump / RECOVERY_MAX_CENTER_JUMP, 1.0)
                        )
                    else:
                        cost = (
                            0.70 * (1.0 - pred_iou)
                            + 0.30 * min(center_jump / RECOVERY_MAX_CENTER_JUMP, 1.0)
                        )

                    if cost <= RECOVERY_MAX_COST:
                        candidates.append((
                            cost, di, lost_track, center_jump,
                            pred_iou, app_dist, gap, False
                        ))

            # One-to-one global greedy assignment. Sort by strongest
            # recovery first so one lost ID cannot be assigned twice.
            candidates.sort(key=lambda x: x[0])
            used_detections = set()
            used_tracks = set()
            recovered_detection_ids = set()

            for (cost, di, lost_track, center_jump,
                 pred_iou, app_dist, gap, hard_reservation) in candidates:

                tid = int(lost_track.track_id)
                if di in used_detections or tid in used_tracks:
                    continue

                # For longer gaps, require a clear recovery. For a short
                # gap, the spatial reservation above is intentionally enough
                # to prevent duplicate/new IDs.
                if not hard_reservation:
                    if app_dist is not None and app_dist > RECOVERY_APP_GATE and pred_iou < 0.20:
                        continue
                    if pred_iou < 0.03 and center_jump > 1.65:
                        continue

                det = detections[di]
                lost_track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(lost_track)

                used_detections.add(di)
                used_tracks.add(tid)
                recovered_detection_ids.add(di)

                print(
                    f"[ID-RECOVERY] frame={self.frame_id} "
                    f"detection={di} -> ID={tid} "
                    f"gap={gap} center={center_jump:.2f} "
                    f"iou={pred_iou:.2f} app="
                    f"{app_dist:.2f}" if app_dist is not None else
                    f"[ID-RECOVERY] frame={self.frame_id} "
                    f"detection={di} -> ID={tid} "
                    f"gap={gap} center={center_jump:.2f} iou={pred_iou:.2f} app=None"
                )

            unmatched_detections = [
                di for di in unmatched_detections
                if di not in recovered_detection_ids
            ]

        # ------------------------------------------------------------
        # CREATE NEW TRACKS ONLY AFTER LOST-TRACK RECOVERY
        # ------------------------------------------------------------

        for inew in unmatched_detections:

            track = detections[inew]

            if track.score < self.det_thresh:
                if vis_type == 'full':
                    init_track_dets_rej.append(track)
                continue

            track.activate(self.kalman_filter, self.frame_id)
            track.update_appearance(getattr(track, 'appearance', None))

            activated_starcks.append(track)

            if vis_type == 'full':
                init_track_dets_acc.append(track)


        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated] # ByteTrack's way

        self.logger.log_local_update_trackets_ids(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
        self.logger.log_state_tracklets_ids(self.tracked_stracks, self.lost_stracks, self.removed_stracks)

        if vis_type == 'full':
            detections_per_assoc_step = {'assoc1': assoc1_dets, 'assoc2': assoc2_dets, 'assoc3': assoc3_dets, 'init_acc': init_track_dets_acc, 'init_rej': init_track_dets_rej}
        else:
            detections_per_assoc_step = None

        removed_tracks_ids = [track.track_id for track in removed_stracks]


        return output_stracks, removed_tracks_ids, new_confirmed_tracks, detections_per_assoc_step, all_considered_tracklets_before_correction


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb