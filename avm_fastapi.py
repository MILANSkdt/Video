"""
AVM - Around View Monitoring System with FastAPI
=================================================
VERSION 9.1 - BUGFIXED Edition

Key fixes in this version:
  * Videos NO LONGER loop – processing stops when the shortest video ends
  * Frame count is read from CAP_PROP_FRAME_COUNT so we know exactly when to stop
  * Fallback: if frame count is unavailable, stops on first failed read (no reset)
  * Output writer is released & flushed before marking job as "done"
  * Temp files are cleaned up after processing
  * Mask normalisation done once at startup, not per-frame
  * Progress logging every 50 frames instead of 100

Two REST APIs:
  POST /api/v1/videoRequest   – Upload 4 videos, stitch, store output
  GET  /api/v1/video/{videoId} – Get video path / stream video

Install:
  pip install fastapi uvicorn python-multipart opencv-python numpy \
              python-jose[cryptography] numba

Run:
  uvicorn avm_fastapi:app --host 0.0.0.0 --port 8000
"""

import cv2
import numpy as np
import os
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from jose import JWTError, jwt

# Numba optional
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARN] Numba not available – using NumPy fallback (slower).")


# =================================================================
#  JWT
# =================================================================
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM  = "HS256"
security   = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired JWT token")


# =================================================================
#  STORAGE
# =================================================================
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\milan.s\API")
TEMP_UPLOAD_DIR    = Path("avm_temp_uploads")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = "avm_config.json"

# Mask file paths – override via environment variables if needed
MASK_FRONT = os.environ.get("MASK_FRONT", "assets/masks/maskFront.jpg")
MASK_BACK  = os.environ.get("MASK_BACK",  "assets/masks/maskBack.jpg")
MASK_LEFT  = os.environ.get("MASK_LEFT",  "assets/masks/maskLeft.jpg")
MASK_RIGHT = os.environ.get("MASK_RIGHT", "assets/masks/maskRight.jpg")
VEHICLE_IMAGE = os.environ.get("VEHICLE_IMAGE", "")

# In-memory job registry
VIDEO_REGISTRY: dict = {}
REGISTRY_LOCK  = threading.Lock()


# =================================================================
#  DIMENSIONS
# =================================================================
PROCESS_SCALE = 0.5

_BASE_CANVAS_W  = 616;  _BASE_CANVAS_H  = 880
_BASE_FRONT_W   = 616;  _BASE_FRONT_H   = 237
_BASE_BACK_W    = 616;  _BASE_BACK_H    = 237
_BASE_LEFT_W    = 218;  _BASE_LEFT_H    = 880
_BASE_RIGHT_W   = 218;  _BASE_RIGHT_H   = 880
_BASE_FRONT_X   = 0;    _BASE_FRONT_Y   = 0
_BASE_BACK_X    = 0;    _BASE_BACK_Y    = 643
_BASE_LEFT_X    = 0;    _BASE_LEFT_Y    = 0
_BASE_RIGHT_X   = 398;  _BASE_RIGHT_Y   = 0
_BASE_VEH_X     = 218;  _BASE_VEH_Y     = 237
_BASE_VEH_W     = 180;  _BASE_VEH_H     = 406

def _s(v): return int(v * PROCESS_SCALE)

CANVAS_W = _s(_BASE_CANVAS_W);  CANVAS_H = _s(_BASE_CANVAS_H)
FRONT_W  = _s(_BASE_FRONT_W);   FRONT_H  = _s(_BASE_FRONT_H)
BACK_W   = _s(_BASE_BACK_W);    BACK_H   = _s(_BASE_BACK_H)
LEFT_W   = _s(_BASE_LEFT_W);    LEFT_H   = _s(_BASE_LEFT_H)
RIGHT_W  = _s(_BASE_RIGHT_W);   RIGHT_H  = _s(_BASE_RIGHT_H)
FRONT_X  = _s(_BASE_FRONT_X);   FRONT_Y  = _s(_BASE_FRONT_Y)
BACK_X   = _s(_BASE_BACK_X);    BACK_Y   = _s(_BASE_BACK_Y)
LEFT_X   = _s(_BASE_LEFT_X);    LEFT_Y   = _s(_BASE_LEFT_Y)
RIGHT_X  = _s(_BASE_RIGHT_X);   RIGHT_Y  = _s(_BASE_RIGHT_Y)
VEH_X    = _s(_BASE_VEH_X);     VEH_Y    = _s(_BASE_VEH_Y)
VEH_W    = _s(_BASE_VEH_W);     VEH_H    = _s(_BASE_VEH_H)

# Full-resolution output dimensions (upscaled back from processing res)
OUT_W = _BASE_CANVAS_W
OUT_H = _BASE_CANVAS_H


# =================================================================
#  FISHEYE PARAMETERS
# =================================================================
FOCAL_LENGTH = 910.0
FISH_SCALE   = 0.5
UNDIS_SCALE  = 1.55
K1 = -0.05611147; K2 = -0.05377447
K3 =  0.01157170; K4 =  0.00307880


# =================================================================
#  CONFIG
# =================================================================
config = {
    "front_dx": 0,  "front_dy": 0,  "back_dx": 0,  "back_dy": 0,
    "left_dx":  0,  "left_dy":  0,  "right_dx": 0, "right_dy": 0,
    "front_scale": 100, "back_scale": 100, "left_scale": 100, "right_scale": 100,
    "front_rot": 0, "back_rot": 0, "left_rot": 0, "right_rot": 0,
    "front_flat": 0, "back_flat": 0, "left_flat": 0, "right_flat": 0,
    "front_crop_top": 0,  "front_crop_bottom": 0,
    "back_crop_top":  0,  "back_crop_bottom":  0,
    "left_crop_top":  0,  "left_crop_bottom":  0,
    "right_crop_top": 0,  "right_crop_bottom": 0,
}

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                config.update(json.load(f))
            print(f"[INFO] Config loaded from {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARN] Could not load config: {e}")


# =================================================================
#  NUMBA / NUMPY IMPLEMENTATIONS
# =================================================================
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True, fastmath=True)
    def _compute_fisheye_maps(out_h, out_w, h, w, focal_length, fish_scale,
                               k1, k2, k3, k4, cx_in, cy_in, cx_out, cy_out):
        map_x = np.empty((out_h, out_w), dtype=np.float32)
        map_y = np.empty((out_h, out_w), dtype=np.float32)
        for oy in prange(out_h):
            for ox in range(out_w):
                nx = (ox - cx_out) / focal_length
                ny = (oy - cy_out) / focal_length
                r  = np.sqrt(nx*nx + ny*ny)
                theta   = np.arctan(r)
                theta_d = theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
                scale   = theta_d / r if r > 1e-8 else 1.0
                map_x[oy, ox] = nx * scale * focal_length / fish_scale + cx_in
                map_y[oy, ox] = ny * scale * focal_length / fish_scale + cy_in
        return map_x, map_y

    @njit(cache=True, fastmath=True)
    def _apply_scale_rot(dst, dst_w, dst_h, scale, rot):
        if scale != 1.0:
            cx = dst_w / 2.0; cy = dst_h / 2.0
            for i in range(4):
                dst[i, 0] = cx + (dst[i, 0] - cx) * scale
                dst[i, 1] = cy + (dst[i, 1] - cy) * scale
        if rot != 0.0:
            angle_rad = np.radians(rot)
            cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
            cx = dst_w / 2.0; cy = dst_h / 2.0
            for i in range(4):
                dx_ = dst[i, 0] - cx; dy_ = dst[i, 1] - cy
                dst[i, 0] = cx + dx_ * cos_a - dy_ * sin_a
                dst[i, 1] = cy + dx_ * sin_a + dy_ * cos_a
        return dst
else:
    def _compute_fisheye_maps(out_h, out_w, h, w, focal_length, fish_scale,
                               k1, k2, k3, k4, cx_in, cy_in, cx_out, cy_out):
        oy, ox = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        nx = (ox - cx_out) / focal_length
        ny = (oy - cy_out) / focal_length
        r  = np.sqrt(nx**2 + ny**2)
        theta   = np.arctan(r)
        theta_d = theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
        with np.errstate(invalid='ignore', divide='ignore'):
            scale = np.where(r > 1e-8, theta_d / r, 1.0)
        return ((nx * scale * focal_length / fish_scale + cx_in).astype(np.float32),
                (ny * scale * focal_length / fish_scale + cy_in).astype(np.float32))

    def _apply_scale_rot(dst, dst_w, dst_h, scale, rot):
        if scale != 1.0:
            c = np.array([dst_w / 2.0, dst_h / 2.0])
            dst = c + (dst - c) * scale
        if rot != 0.0:
            rad  = np.radians(rot)
            R    = np.array([[np.cos(rad), -np.sin(rad)],
                             [np.sin(rad),  np.cos(rad)]])
            c    = np.array([dst_w / 2.0, dst_h / 2.0])
            dst  = np.array([c + R @ (p - c) for p in dst], dtype=np.float32)
        return dst


# =================================================================
#  CORE PIPELINE
# =================================================================
PRE_ROTATE = {
    "front": None,
    "left":  cv2.ROTATE_90_CLOCKWISE,
    "back":  cv2.ROTATE_180,
    "right": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Cache for fisheye maps keyed by (h, w)
_fisheye_cache: dict = {}

def undistort_fisheye(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    key  = (h, w)
    if key not in _fisheye_cache:
        out_w  = int(w * UNDIS_SCALE); out_h = int(h * UNDIS_SCALE)
        map_x, map_y = _compute_fisheye_maps(
            out_h, out_w, h, w, FOCAL_LENGTH, FISH_SCALE,
            K1, K2, K3, K4, w/2.0, h/2.0, out_w/2.0, out_h/2.0
        )
        _fisheye_cache[key] = (map_x, map_y)

    map_x, map_y = _fisheye_cache[key]
    undis = cv2.remap(img, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(undis, cv2.COLOR_BGR2GRAY)
    pts  = cv2.findNonZero(gray)
    if pts is not None:
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = 5
        return undis[max(0,y-pad):min(undis.shape[0],y+bh+pad),
                     max(0,x-pad):min(undis.shape[1],x+bw+pad)]
    return undis


def warp_to_birdseye(img: np.ndarray, label: str,
                     dst_w: int, dst_h: int) -> np.ndarray:
    rot_code = PRE_ROTATE[label]
    if rot_code is not None:
        img = cv2.rotate(img, rot_code)
    if label in ("left", "right"):
        img = cv2.flip(img, 0)

    h, w     = img.shape[:2]
    dx       = config[f"{label}_dx"]
    dy       = config[f"{label}_dy"]
    scale    = config[f"{label}_scale"] / 100.0
    rot      = float(config[f"{label}_rot"])
    flat     = config[f"{label}_flat"] / 100.0
    crop_top = config[f"{label}_crop_top"]
    crop_bot = config[f"{label}_crop_bottom"]

    if crop_top > 0 or crop_bot > 0:
        y1 = crop_top
        y2 = max(h - crop_bot, y1 + 1)
        img = img[y1:y2, :]
        h   = img.shape[0]

    src = np.float32([
        [w*0.05, h*0.05], [w*0.95, h*0.05],
        [w*0.95, h*0.95], [w*0.05, h*0.95],
    ])

    m = 0.02
    compress = 0.15 + 0.35 * flat
    dsts = {
        "front": np.float32([[dst_w*m,            dst_h*m      ],
                              [dst_w*(1-m),        dst_h*m      ],
                              [dst_w*(1-compress), dst_h*(1-m)  ],
                              [dst_w*compress,     dst_h*(1-m)  ]]),
        "left":  np.float32([[dst_w*m,            dst_h*m          ],
                              [dst_w*(1-compress), dst_h*compress   ],
                              [dst_w*(1-compress), dst_h*(1-compress)],
                              [dst_w*m,            dst_h*(1-m)      ]]),
        "back":  np.float32([[dst_w*compress,     dst_h*m      ],
                              [dst_w*(1-compress), dst_h*m      ],
                              [dst_w*(1-m),        dst_h*(1-m)  ],
                              [dst_w*m,            dst_h*(1-m)  ]]),
        "right": np.float32([[dst_w*compress,     dst_h*compress   ],
                              [dst_w*(1-m),        dst_h*m          ],
                              [dst_w*(1-m),        dst_h*(1-m)      ],
                              [dst_w*compress,     dst_h*(1-compress)]]),
    }
    dst = dsts[label].copy()
    if scale != 1.0 or rot != 0.0:
        dst = _apply_scale_rot(dst, float(dst_w), float(dst_h), scale, rot)
    dst[:, 0] += dx
    dst[:, 1] += dy

    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img, H, (dst_w, dst_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def stitch_with_masks(tiles: dict, masks_norm: dict) -> np.ndarray:
    """masks_norm values are float32 in [0, 1], already sized to tile dims."""
    acc    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    weight = np.zeros((CANVAS_H, CANVAS_W),    dtype=np.float32)
    placements = {
        "front": (FRONT_X, FRONT_Y, FRONT_W, FRONT_H),
        "back":  (BACK_X,  BACK_Y,  BACK_W,  BACK_H),
        "left":  (LEFT_X,  LEFT_Y,  LEFT_W,  LEFT_H),
        "right": (RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H),
    }
    for lbl, (px, py, tw, th) in placements.items():
        tile = cv2.resize(tiles[lbl], (tw, th)).astype(np.float32)
        msk  = masks_norm[lbl]                  # already (th, tw) float32
        acc   [py:py+th, px:px+tw] += tile * msk[:, :, np.newaxis]
        weight[py:py+th, px:px+tw] += msk
    w3 = np.maximum(weight[:, :, np.newaxis], 1e-9)
    return np.clip(acc / w3, 0, 255).astype(np.uint8)


def draw_vehicle_silhouette(canvas: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    x0, y0 = VEH_X, VEH_Y
    x1, y1 = VEH_X + VEH_W, VEH_Y + VEH_H
    cx = x0 + VEH_W // 2
    vw, vh = VEH_W, VEH_H
    body=(45,45,45); glass=(170,210,230); roof=(65,65,65)
    wheel=(25,25,25); rim=(90,90,90); line=(100,100,100)
    hood  = np.array([[x0+vw//8,y0+vh//6],[x0+vw//4,y0],[x1-vw//4,y0],[x1-vw//8,y0+vh//6]],np.int32)
    trunk = np.array([[x0+vw//8,y1-vh//6],[x0+vw//4,y1],[x1-vw//4,y1],[x1-vw//8,y1-vh//6]],np.int32)
    cv2.rectangle(out,(x0,y0+vh//6),(x1,y1-vh//6),body,-1)
    cv2.fillPoly(out,[hood],body); cv2.fillPoly(out,[trunk],body)
    rp=np.array([[x0+vw//8,y0+vh//3],[x0+vw//8,y1-vh//3],[x1-vw//8,y1-vh//3],[x1-vw//8,y0+vh//3]],np.int32)
    cv2.fillPoly(out,[rp],roof)
    fg=np.array([[x0+vw//6,y0+vh//6],[x0+vw//4,y0+vh//16],[x1-vw//4,y0+vh//16],[x1-vw//6,y0+vh//6]],np.int32)
    rg=np.array([[x0+vw//6,y1-vh//6],[x0+vw//4,y1-vh//16],[x1-vw//4,y1-vh//16],[x1-vw//6,y1-vh//6]],np.int32)
    cv2.fillPoly(out,[fg],glass); cv2.fillPoly(out,[rg],glass)
    cv2.line(out,(cx,y0),(cx,y1),line,1)
    ww=vw//5; wh=vh//9
    for wx,wy in [(x0-ww//2,y0+vh//8),(x1-ww//2,y0+vh//8),
                  (x0-ww//2,y1-vh//8-wh),(x1-ww//2,y1-vh//8-wh)]:
        cv2.rectangle(out,(wx,wy),(wx+ww,wy+wh),wheel,-1)
        cv2.rectangle(out,(wx,wy),(wx+ww,wy+wh),rim,1)
        cv2.circle(out,(wx+ww//2,wy+wh//2),ww//4,rim,1)
    cv2.rectangle(out,(x0,y0+vh//6),(x1,y1-vh//6),line,1)
    return out


def apply_vehicle_overlay(canvas: np.ndarray) -> np.ndarray:
    if not VEHICLE_IMAGE or not os.path.exists(VEHICLE_IMAGE):
        return draw_vehicle_silhouette(canvas)
    veh = cv2.imread(VEHICLE_IMAGE, cv2.IMREAD_UNCHANGED)
    if veh is None:
        return draw_vehicle_silhouette(canvas)
    veh = cv2.resize(veh, (VEH_W, VEH_H), interpolation=cv2.INTER_LANCZOS4)
    out = canvas.copy()
    if veh.ndim == 3 and veh.shape[2] == 4:
        alpha = veh[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = veh[:, :, :3].astype(np.float32)
        roi   = out[VEH_Y:VEH_Y+VEH_H, VEH_X:VEH_X+VEH_W].astype(np.float32)
        out[VEH_Y:VEH_Y+VEH_H, VEH_X:VEH_X+VEH_W] = (rgb*alpha + roi*(1-alpha)).astype(np.uint8)
    else:
        out[VEH_Y:VEH_Y+VEH_H, VEH_X:VEH_X+VEH_W] = veh[:, :, :3]
    return out


def load_masks() -> dict:
    """Load masks from disk; fall back to all-white if a file is missing."""
    specs = {
        "front": (MASK_FRONT, FRONT_H, FRONT_W),
        "back":  (MASK_BACK,  BACK_H,  BACK_W),
        "left":  (MASK_LEFT,  LEFT_H,  LEFT_W),
        "right": (MASK_RIGHT, RIGHT_H, RIGHT_W),
    }
    masks = {}
    for lbl, (path, eh, ew) in specs.items():
        if os.path.exists(path):
            msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if msk is not None:
                msk = cv2.resize(msk, (ew, eh), interpolation=cv2.INTER_LINEAR)
                masks[lbl] = msk.astype(np.float32) / 255.0
                continue
        print(f"[WARN] Mask not found: {path}  → using uniform white mask for '{lbl}'")
        masks[lbl] = np.ones((eh, ew), dtype=np.float32)
    return masks


# =================================================================
#  VIDEO ID GENERATOR
# =================================================================
def _generate_video_id() -> str:
    return f"vid-{uuid.uuid4().hex[:7]}"


# =================================================================
#  BACKGROUND PROCESSING JOB  ← THE FIXED PART
# =================================================================
def _process_video_job(video_id: str,
                        video_paths: dict,
                        output_dir: Path,
                        masks_norm: dict):
    """
    Reads 4 videos frame-by-frame and writes the stitched AVM output.

    FIX: Stops when ANY camera returns a failed read (no looping).
         Uses CAP_PROP_FRAME_COUNT to know exactly how many frames to
         process so we stop at the shortest video length.
    """
    output_path = output_dir / f"{video_id}.mp4"

    with REGISTRY_LOCK:
        VIDEO_REGISTRY[video_id]["status"] = "processing"

    caps   = {}
    writer = None

    try:
        # ── Open captures ────────────────────────────────────────────
        for lbl, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video '{lbl}': {path}")
            caps[lbl] = cap

        # ── Determine how many frames to process ─────────────────────
        # Use the MINIMUM frame count across all cameras so we stop
        # exactly when the shortest clip ends – no looping, no hang.
        frame_counts = {}
        for lbl, cap in caps.items():
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts[lbl] = fc
            print(f"  [Job {video_id}] {lbl}: {fc} frames reported by OpenCV")

        # Filter out cameras that report 0 or -1 (some containers don't expose it)
        valid_counts = [fc for fc in frame_counts.values() if fc > 0]
        if valid_counts:
            max_frames = min(valid_counts)
            print(f"  [Job {video_id}] Will process {max_frames} frames "
                  f"(shortest valid clip)")
        else:
            max_frames = None   # Unknown – stop on first failed read
            print(f"  [Job {video_id}] Frame count unavailable – "
                  f"will stop on first failed read")

        # ── Video writer setup ────────────────────────────────────────
        fps    = caps["front"].get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (OUT_W, OUT_H))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter could not open output: {output_path}")

        tile_sizes = {
            "front": (FRONT_W, FRONT_H), "back":  (BACK_W,  BACK_H),
            "left":  (LEFT_W,  LEFT_H),  "right": (RIGHT_W, RIGHT_H),
        }

        frame_idx  = 0
        t_start    = time.time()

        # ── Frame loop ────────────────────────────────────────────────
        while True:
            # Stop if we have hit the target frame count
            if max_frames is not None and frame_idx >= max_frames:
                print(f"  [Job {video_id}] Reached {max_frames} frames – stopping.")
                break

            frames = {}
            for lbl, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    # End of this camera's stream → stop everything
                    print(f"  [Job {video_id}] '{lbl}' ended at frame {frame_idx} – stopping.")
                    frames = None
                    break
                # Downscale to processing resolution
                frames[lbl] = cv2.resize(
                    frame,
                    (int(frame.shape[1] * PROCESS_SCALE),
                     int(frame.shape[0] * PROCESS_SCALE)),
                    interpolation=cv2.INTER_AREA,
                )

            if frames is None:
                break   # At least one camera ran out of frames

            # ── Per-frame pipeline ────────────────────────────────────
            tiles = {}
            for lbl in ["front", "back", "left", "right"]:
                undis      = undistort_fisheye(frames[lbl])
                tw, th     = tile_sizes[lbl]
                tiles[lbl] = warp_to_birdseye(undis, lbl, tw, th)

            canvas = stitch_with_masks(tiles, masks_norm)
            final  = apply_vehicle_overlay(canvas)

            # Upscale to full output resolution
            final_full = cv2.resize(final, (OUT_W, OUT_H),
                                    interpolation=cv2.INTER_LINEAR)
            writer.write(final_full)
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = time.time() - t_start
                fps_cur = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  [Job {video_id}] {frame_idx} frames done  "
                      f"({fps_cur:.1f} fps)", flush=True)

        # ── Finalise ──────────────────────────────────────────────────
        writer.release()
        writer = None

        elapsed = time.time() - t_start
        print(f"  [Job {video_id}] Finished – {frame_idx} frames in "
              f"{elapsed:.1f}s → {output_path}")

        with REGISTRY_LOCK:
            VIDEO_REGISTRY[video_id].update({
                "status":      "done",
                "path":        str(output_path.resolve()),
                "frame_count": frame_idx,
                "fps":         fps,
                "elapsed_sec": round(elapsed, 2),
            })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  [Job {video_id}] ERROR: {exc}")
        with REGISTRY_LOCK:
            VIDEO_REGISTRY[video_id].update({"status": "error", "error": str(exc)})

    finally:
        # Always release captures
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        # Always release writer if it wasn't already
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        # Clean up temp uploads
        for path in video_paths.values():
            try:
                os.remove(path)
            except OSError:
                pass
        # Remove empty temp dir
        try:
            temp_dir = Path(list(video_paths.values())[0]).parent
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# =================================================================
#  FASTAPI APPLICATION
# =================================================================
app = FastAPI(
    title="AVM Surround-View API",
    description="Around View Monitoring – fisheye video stitching via REST",
    version="9.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MASKS_NORM: Optional[dict] = None


@app.on_event("startup")
def startup_event():
    global _MASKS_NORM
    load_config()
    _MASKS_NORM = load_masks()
    print(f"[INFO] Canvas: {CANVAS_W}x{CANVAS_H}  →  output: {OUT_W}x{OUT_H}")
    print("[INFO] AVM API v9.1 ready.")


# -----------------------------------------------------------------
#  POST /api/v1/videoRequest
# -----------------------------------------------------------------
@app.post(
    "/api/v1/videoRequest",
    status_code=202,
    summary="Upload 4 fisheye videos and start stitching",
)
async def video_request(
    background_tasks: BackgroundTasks,
    front: UploadFile = File(..., description="Front camera video"),
    back:  UploadFile = File(..., description="Back camera video"),
    left:  UploadFile = File(..., description="Left camera video"),
    right: UploadFile = File(..., description="Right camera video"),
    output_path: Optional[str] = Form(
        None,
        description=(
            "Absolute or relative directory path where the stitched video "
            "will be saved.  Defaults to './avm_processed_videos'."
        ),
    ),
    _token=Depends(verify_token),
):
    """
    **Multipart/form-data** upload of 4 camera videos.

    Returns `{"videoId": "vid-xxxxxxx"}` immediately (HTTP 202).
    Processing runs in the background – poll `/api/v1/video/{videoId}/status`
    or call `GET /api/v1/video/{videoId}` to check when it's done.
    """
    out_dir = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = _generate_video_id()
    temp_dir = TEMP_UPLOAD_DIR / video_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for label, upload in [("front", front), ("back", back),
                           ("left",  left),  ("right", right)]:
        suffix = Path(upload.filename).suffix or ".mp4"
        dest   = temp_dir / f"{label}{suffix}"
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved_paths[label] = str(dest)

    with REGISTRY_LOCK:
        VIDEO_REGISTRY[video_id] = {
            "video_id":   video_id,
            "status":     "queued",
            "path":       None,
            "output_dir": str(out_dir.resolve()),
            "created_at": time.time(),
        }

    background_tasks.add_task(
        _process_video_job,
        video_id,
        saved_paths,
        out_dir,
        _MASKS_NORM,
    )

    print(f"[INFO] Job queued: {video_id}  →  {out_dir}")
    return {"videoId": video_id}


# -----------------------------------------------------------------
#  GET /api/v1/video/{videoId}
# -----------------------------------------------------------------
@app.get(
    "/api/v1/video/{videoId}",
    summary="Get video path and optionally stream the file",
)
def get_video(
    videoId: str,
    stream: bool = False,
    _token=Depends(verify_token),
):
    """
    Retrieve the processed video by **videoId**.

    - While processing: returns `{"status": "processing", "pathToVideo": null}`
    - When done: returns `{"pathToVideo": "<abs path>", ...}` plus a download URL
    - Add `?stream=true` to stream the video bytes directly (for players / browsers)
    """
    with REGISTRY_LOCK:
        entry = VIDEO_REGISTRY.get(videoId)

    if entry is None:
        raise HTTPException(404, detail=f"videoId '{videoId}' not found.")

    status = entry["status"]

    if status in ("queued", "processing"):
        return JSONResponse({
            "videoId":     videoId,
            "status":      status,
            "pathToVideo": None,
            "outputDir":   entry["output_dir"],
        })

    if status == "error":
        raise HTTPException(500, detail={
            "videoId": videoId,
            "status":  "error",
            "error":   entry.get("error", "unknown"),
        })

    # Done
    video_path = Path(entry["path"])
    if not video_path.exists():
        raise HTTPException(404, detail=f"File not found on disk: {video_path}")

    if stream:
        file_size = video_path.stat().st_size

        def _iter_file():
            with open(video_path, "rb") as fh:
                while chunk := fh.read(256 * 1024):
                    yield chunk

        return StreamingResponse(
            _iter_file(),
            media_type="video/mp4",
            headers={
                "Content-Length":      str(file_size),
                "Content-Disposition": f'inline; filename="{video_path.name}"',
                "X-Video-Id":          videoId,
                "X-Path":              str(video_path),
                "Accept-Ranges":       "bytes",
            },
        )

    return JSONResponse({
        "videoId":     videoId,
        "status":      "done",
        "pathToVideo": str(video_path),
        "downloadUrl": f"/api/v1/video/{videoId}?stream=true",
        "frameCount":  entry.get("frame_count"),
        "fps":         entry.get("fps"),
        "elapsedSec":  entry.get("elapsed_sec"),
    })


# -----------------------------------------------------------------
#  GET /api/v1/video/{videoId}/status  (lightweight poll)
# -----------------------------------------------------------------
@app.get("/api/v1/video/{videoId}/status", summary="Poll processing status")
def get_video_status(videoId: str, _token=Depends(verify_token)):
    with REGISTRY_LOCK:
        entry = VIDEO_REGISTRY.get(videoId)
    if entry is None:
        raise HTTPException(404, detail=f"videoId '{videoId}' not found.")
    return {
        "videoId":     videoId,
        "status":      entry["status"],
        "pathToVideo": entry.get("path"),
        "createdAt":   entry.get("created_at"),
        "frameCount":  entry.get("frame_count"),
    }


# -----------------------------------------------------------------
#  DEV ONLY: generate a test JWT  (remove in production)
# -----------------------------------------------------------------
@app.get("/api/v1/dev/token", include_in_schema=False)
def dev_token():
    token = jwt.encode(
        {"sub": "dev-user", "exp": int(time.time()) + 3600},
        SECRET_KEY, algorithm=ALGORITHM,
    )
    return {"token": token, "note": "Development only – remove before deploying"}


# =================================================================
#  ENTRY POINT
# =================================================================
if __name__ == "__main__":
    uvicorn.run(
        "avm_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,   # Keep 1 – Numba JIT cache is process-local
    )