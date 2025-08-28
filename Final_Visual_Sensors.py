"""
px4_agent_visual_node: A ROS2 node for visual perception and object tracking.

This script subscribes to RGB and depth images published by Gazebo/ROS2 and runs a
simple object detection pipeline.  Detected objects are annotated on a live
OpenCV window with class labels and estimated 3D positions (X, Y, Z) derived
from the depth image and camera intrinsics.  Results are printed to the
terminal at a configurable rate, logged to a JSON Lines file, published as a
Detection3DArray on a ROS2 topic (when vision_msgs is available), and served
via a lightweight HTTP API for easy integration with other components (for
example, a PX4 offboard control agent).

Key features:
 - Automatically resizes or maps RGB and depth images if their resolutions
   differ, ensuring depth lookups correspond to the correct pixel in the RGB
   image.
 - Attempts to use a YOLOv8 model via the ultralytics package if available;
   falls back to OpenCV's HOG-based person detector when ultralytics or
   weights are missing.  This ensures the node runs without crashing even
   when no external models are installed.
 - Assigns a unique colour to each detected class for improved visual
   distinction.
 - Computes real‑world X, Y, Z positions from pixel coordinates and depth
   values using intrinsic parameters from `/camera_info`.  If intrinsics
   haven't been received yet, sensible defaults based on the image size and
   a typical field of view are used.
 - Logs each frame's detections to a JSONL file with timestamps for later
   analysis or dataset generation.
 - Provides an HTTP endpoint that returns the latest detections and frame
   statistics in JSON format.

Note: This script relies on ROS2 (rclpy), OpenCV, numpy and optionally
ultralytics.  It should be run in an environment where ROS2 is available.

Usage:
  python code.py

No command line arguments are required – all configurable parameters are
exposed via ROS2 parameters with sensible defaults.
"""

import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

import collections  # For deque to track last few detection frames

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from builtin_interfaces.msg import Time as RosTime
    try:
        from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
        from geometry_msgs.msg import Pose, Point, Quaternion
        VISION_MSGS_AVAILABLE = True
    except Exception:
        VISION_MSGS_AVAILABLE = False
    try:
        from cv_bridge import CvBridge  # Optional, if available
        CV_BRIDGE_AVAILABLE = True
    except Exception:
        CV_BRIDGE_AVAILABLE = False
    try:
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        MESSAGE_FILTERS_AVAILABLE = True
    except Exception:
        MESSAGE_FILTERS_AVAILABLE = False
except Exception:
    # If rclpy isn't available, print a message but allow the script to be
    # imported for static analysis.  Running without rclpy will do nothing.
    rclpy = None
    Node = object  # type: ignore
    Image = CameraInfo = object  # type: ignore
    Detection3DArray = Detection3D = ObjectHypothesisWithPose = Pose = Point = Quaternion = object  # type: ignore
    VISION_MSGS_AVAILABLE = False
    CV_BRIDGE_AVAILABLE = False
    MESSAGE_FILTERS_AVAILABLE = False

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    # Prefer ultralytics if available for robust detection across many classes.
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# Store the JSONL file alongside this script by default.  Using the
# directory of the current file makes it straightforward for users to
# locate the JSON log regardless of their working directory when
# launching the script.
try:
    # __file__ is defined when running as a script; fall back to cwd if not
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except Exception:
    _script_dir = os.getcwd()
DEFAULT_JSONL_PATH = os.path.join(_script_dir, "detections_log.jsonl")

# Global store for HTTP server to retrieve the latest detections.
#
# `latest_detection_data` holds the most recent frame's detection summary for
# backwards compatibility.  `recent_detection_history` holds the last three
# detection frames for agents that need a rolling window of detections.
latest_detection_data: Dict[str, Any] = {
    "timestamp": None,
    "detections": [],
    "fps": 0.0,
    "detector": ""
}

# Store the last three detection records.  Each entry is a dict with keys
# 'timestamp', 'detector', 'fps' and 'detections'.  This history is updated
# by the VisualPerceptionNode and served via the HTTP API.
recent_detection_history: List[Dict[str, Any]] = []


def load_default_colors() -> Dict[str, Tuple[int, int, int]]:
    """Generate a deterministic colour map for the first few classes.

    The ultralytics YOLO model comes with a predefined list of class names.
    If those aren't available (e.g. using HOG), we'll still assign random
    colours to any class names encountered.
    """
    colours = {}
    random.seed(42)
    # Predefine some common classes with pleasant colours
    predefined = {
        "person": (255, 0, 0),  # Red
        "car": (0, 255, 0),     # Green
        "truck": (0, 0, 255),   # Blue
        "bicycle": (255, 255, 0),  # Cyan
        "motorcycle": (255, 0, 255),  # Magenta
        "bus": (0, 255, 255),  # Yellow
        "train": (128, 0, 128),  # Purple
        "dog": (128, 128, 0),  # Olive
        "cat": (0, 128, 128),  # Teal
        "bird": (128, 0, 0),  # Maroon
    }
    colours.update(predefined)
    return colours


class SimpleHTTPDetectionHandler(BaseHTTPRequestHandler):
    """A basic HTTP handler that returns the latest detection data."""

    def do_GET(self) -> None:
        if self.path.startswith("/detections"):
            # Return the latest detection data as JSON
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            # Serve only the recent detection history (last few entries) in a
            # human‑friendly, vertically formatted JSON.  We return the
            # formatted history as produced by VisualPerceptionNode.
            try:
                response = {
                    "history": recent_detection_history
                }
                self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
            except Exception:
                # If something goes wrong, send an empty history
                self.wfile.write(json.dumps({"history": []}, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress HTTP log messages to avoid cluttering terminal
        return


class VisualPerceptionNode(Node):
    """
    ROS2 node that subscribes to camera and depth topics, runs object detection,
    computes 3D positions, displays annotated images, logs results, publishes
    detections, and provides an HTTP endpoint.
    """

    def __init__(self) -> None:
        super().__init__("px4_agent_visual_node")
        self.get_logger().info("Initializing Visual Perception Node")

        # Parameters with sensible defaults.  These can be changed via
        # `ros2 param set` or during node launch.
        self.declare_parameter("camera_topic", "/camera")
        self.declare_parameter("depth_topic", "/depth_camera")
        self.declare_parameter("camera_info_topic", "/camera_info")
        # Publishing and logging options
        self.declare_parameter("publish_detections_3d", True)
        self.declare_parameter("output_topic_3d", "/detected_objects_3d")
        self.declare_parameter("jsonl_enabled", True)
        self.declare_parameter("jsonl_path", DEFAULT_JSONL_PATH)
        # Visualisation and printing
        self.declare_parameter("show_window", True)
        self.declare_parameter("print_hz", 2.0)
        self.declare_parameter("depth_patch", 5)
        self.declare_parameter("min_depth_m", 0.1)
        self.declare_parameter("max_depth_m", 60.0)
        self.declare_parameter("overlay_scale", 1.0)
        # How many seconds of history to keep.  This controls the length of
        # the detection history used for logging and API serving.  Increase
        # this value to retain more past detections (e.g., 7 seconds).
        self.declare_parameter("history_seconds", 5)
        # HTTP server parameters
        self.declare_parameter("http_enabled", True)
        self.declare_parameter("http_port", 8088)

        # Read parameters
        self.camera_topic: str = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.depth_topic: str = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic: str = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.publish_detections_3d: bool = self.get_parameter("publish_detections_3d").get_parameter_value().bool_value
        self.output_topic_3d: str = self.get_parameter("output_topic_3d").get_parameter_value().string_value
        self.jsonl_enabled: bool = self.get_parameter("jsonl_enabled").get_parameter_value().bool_value
        self.jsonl_path: str = self.get_parameter("jsonl_path").get_parameter_value().string_value
        self.show_window: bool = self.get_parameter("show_window").get_parameter_value().bool_value
        self.print_hz: float = self.get_parameter("print_hz").get_parameter_value().double_value
        self.depth_patch: int = self.get_parameter("depth_patch").get_parameter_value().integer_value
        self.min_depth_m: float = self.get_parameter("min_depth_m").get_parameter_value().double_value
        self.max_depth_m: float = self.get_parameter("max_depth_m").get_parameter_value().double_value
        self.overlay_scale: float = self.get_parameter("overlay_scale").get_parameter_value().double_value
        self.http_enabled: bool = self.get_parameter("http_enabled").get_parameter_value().bool_value
        self.http_port: int = self.get_parameter("http_port").get_parameter_value().integer_value

        # History length: number of seconds to store detection entries.
        self.history_seconds: int = max(1, self.get_parameter("history_seconds").get_parameter_value().integer_value)

        # ------------------------------------------------------------------
        # Summary tracking configuration
        #
        # Unlike the raw detection history (which stores every processed frame),
        # we maintain a higher level summary of the most interesting detection
        # once per second.  Each summary entry captures the nearest (or most
        # confident) object at the time of summarisation along with its
        # bounding box in world coordinates, centre point, depth and
        # confidence.  The number of entries retained is controlled by
        # `history_seconds`, allowing users to request the last N seconds of
        # detections via the API or JSON file.  A summary counter provides
        # human‑friendly IDs (Entry 001, Entry 002, etc.).
        self.summary_interval: float = 1.0  # seconds between summaries
        self.summary_history_len: int = self.history_seconds  # number of summaries to keep
        self.summaries: collections.deque = collections.deque(maxlen=self.summary_history_len)
        self.summary_counter: int = 0
        self.last_summary_time: float = time.time()

        # Camera intrinsics (fx, fy, cx, cy) and defaults
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Colour map per class
        self.colours: Dict[str, Tuple[int, int, int]] = load_default_colors()

        # Bridge for converting ROS images to CV images (if available)
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None

        # Initialise detection model
        self.detector_name = ""
        self.detector = None
        if ULTRALYTICS_AVAILABLE:
            try:
                # Try to load YOLOv8s (small) model; will download weights on first use
                self.detector = YOLO("yolov8s.pt")
                self.detector_name = "YOLOv8s"
                self.get_logger().info("Loaded ultralytics YOLOv8s model for detection")
            except Exception as e:
                self.get_logger().warn(f"YOLOv8 failed to load: {e}\nFalling back to HOG detector.")
                self.detector = None
        if self.detector is None:
            # Fallback: HOG person detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.detector = hog
            self.detector_name = "HOG"
            self.get_logger().info("Using HOG person detector as fallback")

        # Rolling buffer of recent detection records.  The buffer length
        # corresponds to the number of seconds of history to retain when
        # writing to JSON and serving via the API.  It is set based on
        # `history_seconds` so users can change it (e.g., to 7 seconds).
        self.recent_records: collections.deque = collections.deque(maxlen=self.history_seconds)
        # JSON file handle is no longer held open; we write the last 3 records
        # each time new detections are processed.  We'll only hold the path and
        # flag for saving.
        self.jsonl_file = None
        if self.jsonl_enabled:
            # Attempt to create the file if it doesn't exist.  We don't keep
            # the handle open because the file is rewritten on each update.
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
                # Create or truncate file
                with open(self.jsonl_path, "w") as f:
                    pass
                # Note: we'll keep the last 5 entries in the JSON file,
                # matching the length of `recent_records`.
                # Inform user about history retention: this file will contain the
                # most recent `history_seconds` summary entries, one per
                # second.
                self.get_logger().info(
                    f"Will store the last {self.history_seconds} summaries in {self.jsonl_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to initialise JSON file: {e}")
                self.jsonl_enabled = False

        # Publisher for 3D detections
        self.det_3d_pub = None
        if self.publish_detections_3d and VISION_MSGS_AVAILABLE:
            self.det_3d_pub = self.create_publisher(Detection3DArray, self.output_topic_3d, 10)
        elif self.publish_detections_3d:
            self.get_logger().warn("vision_msgs not available; 3D detection publishing disabled")
            self.publish_detections_3d = False

        # Subscribers and synchroniser
        if MESSAGE_FILTERS_AVAILABLE:
            self.create_subscribers_with_sync()
        else:
            # Fallback: simple subscriptions without exact synchronisation
            self.get_logger().warn("message_filters unavailable; using approximate callback sequencing")
            self.camera_sub = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
            self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
            self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
            # Buffers for latest images
            self.latest_rgb: Optional[Image] = None
            self.latest_depth: Optional[Image] = None
            # Timer to process frames at a reasonable rate
            self.timer = self.create_timer(0.1, self.process_latest_frames)

        # Timers and counters
        self.last_print_time = time.time()
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Track the last time we wrote the detection history to the JSON
        # file.  We'll only update the file and the API once per second.
        self.last_json_update_time = time.time()

        # Optionally start HTTP server
        if self.http_enabled:
            threading.Thread(target=self.start_http_server, daemon=True).start()

    # ------------------------------------------------------------------
    # Subscription setup using message_filters if available
    def create_subscribers_with_sync(self) -> None:
        """Create message_filters subscribers and synchroniser."""
        self.get_logger().info("Setting up subscribers with ApproximateTimeSynchronizer")
        # Use separate subscribers for each topic
        cam_sub = Subscriber(self, Image, self.camera_topic)
        depth_sub = Subscriber(self, Image, self.depth_topic)
        info_sub = Subscriber(self, CameraInfo, self.camera_info_topic)
        sync = ApproximateTimeSynchronizer([cam_sub, depth_sub, info_sub], queue_size=30, slop=0.1)
        sync.registerCallback(self.synced_callback)

    # ------------------------------------------------------------------
    # Callbacks for synchronised messages
    def synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        """Handle synchronised RGB, depth and camera info messages."""
        self.process_frame(rgb_msg, depth_msg, info_msg)

    # ------------------------------------------------------------------
    # Fallback subscribers: handle independently
    def camera_callback(self, msg: Image) -> None:
        self.latest_rgb = msg

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.update_intrinsics(msg)

    def process_latest_frames(self) -> None:
        """Process latest buffered frames when not using message_filters."""
        if self.latest_rgb and self.latest_depth:
            # Copy references and clear buffers to avoid processing same frame again
            rgb_msg = self.latest_rgb
            depth_msg = self.latest_depth
            self.latest_rgb = None
            self.latest_depth = None
            # Use the latest intrinsics we have
            if self.fx is None:
                # We still try to update intrinsics lazily if info has not been received
                pass
            dummy_info = CameraInfo()
            dummy_info.k = [self.fx or 0.0, 0.0, self.cx or 0.0,
                            0.0, self.fy or 0.0, self.cy or 0.0,
                            0.0, 0.0, 1.0]
            self.process_frame(rgb_msg, depth_msg, dummy_info)

    # ------------------------------------------------------------------
    # Intrinsics update from camera_info
    def update_intrinsics(self, msg: CameraInfo) -> None:
        """Extract focal lengths and principal point from camera_info message."""
        # Only update once to avoid jitter if multiple camera_info messages arrive
        if not self.fx or not self.fy:
            try:
                # The matrix is [fx 0 cx; 0 fy cy; 0 0 1]
                self.fx = float(msg.k[0]) if msg.k[0] else None
                self.fy = float(msg.k[4]) if msg.k[4] else None
                self.cx = float(msg.k[2]) if msg.k[2] else None
                self.cy = float(msg.k[5]) if msg.k[5] else None
                if self.fx and self.fy:
                    self.get_logger().info(f"Updated camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            except Exception as e:
                self.get_logger().warn(f"Failed to parse camera intrinsics: {e}")

    # ------------------------------------------------------------------
    # Frame processing
    def process_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        """Main image processing pipeline: runs detection, computes 3D, displays and logs."""
        # Convert ROS Image messages to numpy arrays
        rgb_image = self.rosimg_to_cv2(rgb_msg)
        depth_image, depth_width, depth_height = self.rosimg_depth_to_numpy(depth_msg)

        if rgb_image is None or depth_image is None:
            return

        # Update intrinsics from info if available
        if info_msg is not None and (not self.fx or not self.fy):
            self.update_intrinsics(info_msg)

        rgb_height, rgb_width = rgb_image.shape[:2]

        # Detection: run YOLO or HOG
        detections = self.run_detection(rgb_image)

        # Compute FPS
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_fps_time = now

        # Prepare output list
        detections_out = []
        det3d_msgs = []  # For vision_msgs
        # Process detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["label"]
            score = det.get("score", 0.0)
            # Compute center in RGB image
            u_rgb = int((x1 + x2) / 2)
            v_rgb = int((y1 + y2) / 2)
            # Map to depth image coordinates
            depth_u, depth_v = self.map_rgb_to_depth(u_rgb, v_rgb, rgb_width, rgb_height, depth_width, depth_height)
            # Extract depth patch
            patch_size = self.depth_patch
            half = patch_size // 2
            x0 = max(0, depth_u - half)
            y0 = max(0, depth_v - half)
            x1p = min(depth_width, depth_u + half + 1)
            y1p = min(depth_height, depth_v + half + 1)
            patch = depth_image[y0:y1p, x0:x1p]
            # Filter out invalid values
            valid = patch[np.isfinite(patch)]
            valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]
            if valid.size == 0:
                Z = float("nan")
            else:
                Z = float(np.median(valid))
            # Compute X, Y from intrinsics
            if self.fx and self.fy and self.cx is not None and self.cy is not None and not np.isnan(Z):
                X = (float(depth_u) - self.cx) * Z / self.fx
                Y = (float(depth_v) - self.cy) * Z / self.fy
            else:
                # Approximate XY assuming centre principal point and focal lengths based on image width
                # Use typical HFOV 60 degrees (1.047 rad) if unknown
                f_approx = rgb_width / (2 * np.tan(1.047 / 2))
                cx_approx = rgb_width / 2.0
                cy_approx = rgb_height / 2.0
                X = (float(depth_u) - cx_approx) * Z / f_approx
                Y = (float(depth_v) - cy_approx) * Z / f_approx
            # Save detection result
            det_out = {
                "label": cls_name,
                "score": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_rgb": [u_rgb, v_rgb],
                "uv_depth": [depth_u, depth_v],
                "position": {"x": X, "y": Y, "z": Z},
            }
            detections_out.append(det_out)
            # Prepare vision_msgs if needed
            if self.publish_detections_3d:
                det3d = Detection3D()
                det3d.header = self.create_header(rgb_msg.header.stamp, rgb_msg.header.frame_id)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = cls_name
                hyp.hypothesis.score = float(score)
                hyp.pose = Pose()
                hyp.pose.position = Point(x=X, y=Y, z=Z)
                hyp.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                det3d.results.append(hyp)
                det3d_msgs.append(det3d)

        # Publish 3D detections if applicable
        if self.publish_detections_3d and det3d_msgs:
            det_array = Detection3DArray()
            det_array.header = self.create_header(rgb_msg.header.stamp, rgb_msg.header.frame_id)
            det_array.detections = det3d_msgs
            self.det_3d_pub.publish(det_array)

        # Build a record for this frame containing all detections and metadata
        record = {
            "timestamp": time.time(),
            "detector": self.detector_name,
            "fps": self.fps,
            "detections": detections_out,
        }
        # Append the raw record to our rolling buffer of frames
        self.recent_records.append(record)

        # ------------------------------------------------------------------
        # Summary update: generate a high‑level summary once every
        # `self.summary_interval` seconds.  Each summary captures the
        # nearest detection (by Z distance) and records its bounding box,
        # centre, depth and confidence in world coordinates.
        now_time = time.time()
        if now_time - self.last_summary_time >= self.summary_interval:
            # Determine the detection to summarise: choose the one with the
            # smallest positive Z (nearest to the camera).  If no valid
            # detections are present, record a placeholder entry.
            summary_entry: Dict[str, Any] = {}
            chosen_det: Optional[Dict[str, Any]] = None
            min_dist: float = float('inf')
            for det in detections_out:
                z_val = det["position"].get("z")
                if z_val is not None and not np.isnan(z_val) and z_val > 0 and z_val < min_dist:
                    min_dist = z_val
                    chosen_det = det
            # Increment ID counter
            self.summary_counter += 1
            # Use a simple ID prefix (e.g. "ID 001") for each summary entry
            entry_id = f"ID {self.summary_counter:03d}"
            # If a detection exists, compute world bounding box and centre
            if chosen_det:
                # Compute world bounding box corners using the same depth
                # as the detection centre.  This approximates the extent
                # of the object in the horizontal (X) and vertical (Y)
                # directions.  For each pixel corner, project to world
                # coordinates using intrinsics (or fall back to approximate).
                bbox_px = chosen_det.get("bbox", [0, 0, 0, 0])
                xmin_px, ymin_px, xmax_px, ymax_px = bbox_px
                u_center, v_center = chosen_det.get("uv_depth", [0, 0])
                Z = chosen_det["position"].get("z", float('nan'))
                # Use intrinsics if available
                if self.fx and self.fy and self.cx is not None and self.cy is not None and not np.isnan(Z):
                    # Project each corner
                    Xmin = (float(xmin_px) - self.cx) * Z / self.fx
                    Xmax = (float(xmax_px) - self.cx) * Z / self.fx
                    Ymin = (float(ymin_px) - self.cy) * Z / self.fy
                    Ymax = (float(ymax_px) - self.cy) * Z / self.fy
                    # Centre
                    Xc = chosen_det["position"].get("x")
                    Yc = chosen_det["position"].get("y")
                else:
                    # Approximate projection if intrinsics missing
                    # Use approximate focal length based on RGB width and 60° HFOV
                    f_approx = rgb_width / (2 * np.tan(1.047 / 2))
                    cx_approx = rgb_width / 2.0
                    cy_approx = rgb_height / 2.0
                    Xmin = (float(xmin_px) - cx_approx) * Z / f_approx
                    Xmax = (float(xmax_px) - cx_approx) * Z / f_approx
                    Ymin = (float(ymin_px) - cy_approx) * Z / f_approx
                    Ymax = (float(ymax_px) - cy_approx) * Z / f_approx
                    Xc = chosen_det["position"].get("x")
                    Yc = chosen_det["position"].get("y")
                # Build bounding box and centre dictionaries with raw float values.  We
                # defer formatting to the JSON writer so that numbers can be
                # rendered consistently with two decimals.  If a value is NaN,
                # leave it as None for easier downstream handling.
                bbox_dict: Dict[str, Optional[float]] = {
                    "Xmin": Xmin if not np.isnan(Xmin) else None,
                    "Xmax": Xmax if not np.isnan(Xmax) else None,
                    "Ymin": Ymin if not np.isnan(Ymin) else None,
                    "Ymax": Ymax if not np.isnan(Ymax) else None,
                }
                centre_dict: Dict[str, Optional[float]] = {
                    "Xcenter": Xc if Xc is not None and not np.isnan(Xc) else None,
                    "Ycenter": Yc if Yc is not None and not np.isnan(Yc) else None,
                }
                # Build the summary entry using raw numbers; these will
                # be formatted later when writing the JSON/API output.
                summary_entry = {
                    "ID": entry_id,
                    "time": now_time,
                    "Object Name": chosen_det.get("label", ""),
                    "Bounding Box": bbox_dict,
                    "Center Point": centre_dict,
                    "Depth": round(Z, 2) if not np.isnan(Z) else None,
                    "Confidence Level": round(chosen_det.get("score", 0.0), 2),
                }
            else:
                # No detection: record an empty entry
                summary_entry = {
                    "ID": entry_id,
                    "time": now_time,
                    "Object Name": "none",
                    "Bounding Box": {},
                    "Center Point": {},
                    "Depth": None,
                    "Confidence Level": None,
                }
            # Append summary to deque
            self.summaries.append(summary_entry)
            # Update summary timer
            self.last_summary_time = now_time

        # Update the global latest detection data (for compatibility)
        latest_detection_data["timestamp"] = record["timestamp"]
        latest_detection_data["detections"] = record["detections"]
        latest_detection_data["fps"] = record["fps"]
        latest_detection_data["detector"] = record["detector"]
        # Update JSON file and API history if one second has passed since the last update
        self.update_json_and_api()

        # Display window
        if self.show_window:
            self.draw_and_show(rgb_image.copy(), detections_out, rgb_width, rgb_height)

        # Print periodic summary
        self.print_periodic_summary(detections_out)

        # Append to JSONL
        if self.jsonl_file and detections_out:
            record = {
                "timestamp": time.time(),
                "detector": self.detector_name,
                "detections": detections_out,
                "fps": self.fps,
            }
            self.jsonl_file.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Helper: convert ROS Image to CV2
    def rosimg_to_cv2(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if CV_BRIDGE_AVAILABLE and isinstance(msg.data, (bytes, bytearray)):
                # Use cv_bridge when available
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')  # type: ignore[arg-type]
            else:
                # Manual conversion: assume encoding rgb8 or bgr8
                dtype = np.uint8
                # Determine number of channels from encoding
                if msg.encoding.lower() in ["rgb8", "bgr8"]:
                    channels = 3
                elif msg.encoding.lower() in ["mono8"]:
                    channels = 1
                else:
                    self.get_logger().warn(f"Unsupported encoding {msg.encoding}")
                    return None
                img = np.frombuffer(msg.data, dtype=dtype)
                img = img.reshape((msg.height, msg.width, channels))
                if msg.encoding.lower() == "bgr8":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert RGB image: {e}")
            return None

    # ------------------------------------------------------------------
    # Helper: convert depth image to numpy
    def rosimg_depth_to_numpy(self, msg: Image) -> Tuple[Optional[np.ndarray], int, int]:
        try:
            # Depth encoding 32FC1 -> float32 per pixel
            if msg.encoding in ["32FC1", "32fc1", "Type_32FC1", "32FC1"]:
                array = np.frombuffer(msg.data, dtype=np.float32)
                array = array.reshape((msg.height, msg.width))
                return array, msg.width, msg.height
            else:
                # Attempt to interpret uint16 as depth in millimetres
                dtype = np.uint16 if msg.encoding.lower() in ["16uc1", "type_16uc1", "uint16"] else np.float32
                array = np.frombuffer(msg.data, dtype=dtype)
                array = array.reshape((msg.height, msg.width))
                if dtype == np.uint16:
                    array = array.astype(np.float32) * 0.001  # convert mm to metres
                return array, msg.width, msg.height
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")
            return None, 0, 0

    # ------------------------------------------------------------------
    # Helper: map RGB pixel to depth pixel coordinates
    def map_rgb_to_depth(self, u_rgb: int, v_rgb: int, rgb_w: int, rgb_h: int, depth_w: int, depth_h: int) -> Tuple[int, int]:
        if rgb_w == depth_w and rgb_h == depth_h:
            return u_rgb, v_rgb
        # Compute scaling factors
        sx = depth_w / float(rgb_w)
        sy = depth_h / float(rgb_h)
        u_d = int(u_rgb * sx)
        v_d = int(v_rgb * sy)
        # Clamp
        u_d = max(0, min(depth_w - 1, u_d))
        v_d = max(0, min(depth_h - 1, v_d))
        return u_d, v_d

    # ------------------------------------------------------------------
    # Detection wrapper
    def run_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on the RGB image and return a list of detections."""
        detections: List[Dict[str, Any]] = []
        if self.detector_name == "YOLOv8s" and self.detector is not None:
            try:
                # YOLOv8 requires BGR input; convert if necessary
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                results = self.detector(bgr)
                # Each result contains boxes, classes, etc.
                for res in results:
                    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
                    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else []
                    cls_ids = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, 'cls') else []
                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.detector.model.names[int(cls_id)] if hasattr(self.detector.model, 'names') else str(int(cls_id))
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": class_name,
                            "score": float(conf),
                        })
            except Exception as e:
                self.get_logger().warn(f"YOLO detection failed: {e}")
        elif self.detector_name == "HOG" and self.detector is not None:
            # HOG only detects people; returns rects as x,y,w,h
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                rects, weights = self.detector.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
                for (x, y, w, h), score in zip(rects, weights):
                    detections.append({
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "label": "person",
                        "score": float(score[0] if hasattr(score, '__len__') else score),
                    })
            except Exception as e:
                self.get_logger().warn(f"HOG detection failed: {e}")
        else:
            # No detector: return empty list
            pass
        return detections

    # ------------------------------------------------------------------
    # Draw overlay and show window
    def draw_and_show(self, image: np.ndarray, detections: List[Dict[str, Any]], width: int, height: int) -> None:
        scale = self.overlay_scale
        # If overlay_scale != 1, resize for window display
        if scale != 1.0:
            image_display = cv2.resize(image, (int(width * scale), int(height * scale)))
        else:
            image_display = image
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Scale coordinates if display size differs
            if scale != 1.0:
                x1d, y1d, x2d, y2d = [int(coord * scale) for coord in (x1, y1, x2, y2)]
            else:
                x1d, y1d, x2d, y2d = x1, y1, x2, y2
            cls_name = det["label"]
            pos = det["position"]
            # Assign colour for this class
            colour = self.colours.get(cls_name)
            if colour is None:
                # Generate a random colour if not predefined
                colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self.colours[cls_name] = colour
            # Draw rectangle
            cv2.rectangle(image_display, (x1d, y1d), (x2d, y2d), colour, 2)
            # Compose label text with position
            x_str = f"{pos['x']:.2f}m" if not np.isnan(pos['x']) else "NaN"
            y_str = f"{pos['y']:.2f}m" if not np.isnan(pos['y']) else "NaN"
            z_str = f"{pos['z']:.2f}m" if not np.isnan(pos['z']) else "NaN"
            label = f"{cls_name}: X={x_str}, Y={y_str}, Z={z_str}"
            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image_display, (x1d, y1d - text_h - baseline), (x1d + text_w, y1d), colour, -1)
            cv2.putText(image_display, label, (x1d, y1d - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        # Draw FPS on corner
        fps_text = f"FPS: {self.fps:.1f}" if self.fps else "FPS: ..."
        cv2.putText(image_display, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        try:
            cv2.imshow("PX4 Agent ? Visual Perception", image_display)
            cv2.waitKey(1)
        except Exception:
            # If display isn't available (e.g. headless), disable show_window
            self.show_window = False

    # ------------------------------------------------------------------
    # Create a ROS header from stamp
    def create_header(self, stamp: RosTime, frame_id: str) -> 'Header':  # type: ignore[name-defined]
        hdr = type('Header', (), {})()
        hdr.stamp = stamp
        hdr.frame_id = frame_id
        return hdr

    # ------------------------------------------------------------------
    # Periodic terminal output
    def print_periodic_summary(self, detections: List[Dict[str, Any]]) -> None:
        now = time.time()
        if now - self.last_print_time >= (1.0 / max(self.print_hz, 0.001)):
            count = len(detections)
            nearest = None
            nearest_dist = float('inf')
            for det in detections:
                z = det['position']['z']
                if not np.isnan(z) and z < nearest_dist:
                    nearest_dist = z
                    nearest = det
            nearest_str = f"{nearest['label']} @ Z={nearest_dist:.2f}m" if nearest else "none"
            self.get_logger().info(f"[Detections] count={count}, nearest: {nearest_str}, FPS={self.fps:.1f}")
            self.last_print_time = now

    # ------------------------------------------------------------------
    # HTTP server thread
    def start_http_server(self) -> None:
        self.get_logger().info(f"Starting HTTP server on port {self.http_port}")
        server = HTTPServer(('0.0.0.0', self.http_port), SimpleHTTPDetectionHandler)
        try:
            server.serve_forever()
        except Exception as e:
            self.get_logger().warn(f"HTTP server error: {e}")

    # ------------------------------------------------------------------
    # Update JSON file and global API state
    def update_json_and_api(self) -> None:
        """
        Serialise the most recent summary records to a JSON file and
        update the global detection history for the HTTP API.  This method
        writes the summary history at most once per second to avoid
        excessive file I/O and API churn.  Each summary entry includes
        a time delta (seconds ago) with two decimal places and details
        about the selected object (ID, name, bounding box, centre,
        depth and confidence).  Numbers are formatted with two decimal
        places for readability.
        """
        now = time.time()
        # Only update if at least one second has passed since the last write
        if now - self.last_json_update_time < 1.0:
            return
        # Build formatted summaries based on self.summaries
        formatted_history: List[Dict[str, Any]] = []
        for entry in list(self.summaries):
            # Compute a human‑readable time string: use local time of the
            # summary creation moment.  Format with two decimal places for
            # seconds.  If timezone information is unavailable, assume
            # system local time.
            ts = entry.get("time", now)
            try:
                dt_obj = datetime.fromtimestamp(ts)
                time_str = dt_obj.strftime("%H:%M:%S.%f")[:-4]  # HH:MM:SS.xx
            except Exception:
                # Fallback to seconds ago if conversion fails
                delta_sec = max(0.0, now - ts)
                time_str = f"{delta_sec:.2f}"
            # Prepare bounding box and centre; format numbers with two decimals
            bbox = entry.get("Bounding Box", {})
            centre = entry.get("Center Point", {})
            bb_json = {
                "Xmin": f"{bbox.get('Xmin'):.2f}" if bbox.get("Xmin") is not None else None,
                "Xmax": f"{bbox.get('Xmax'):.2f}" if bbox.get("Xmax") is not None else None,
                "Ymin": f"{bbox.get('Ymin'):.2f}" if bbox.get("Ymin") is not None else None,
                "Ymax": f"{bbox.get('Ymax'):.2f}" if bbox.get("Ymax") is not None else None,
            }
            centre_json = {
                "Xcenter": f"{centre.get('Xcenter'):.2f}" if centre.get("Xcenter") is not None else None,
                "Ycenter": f"{centre.get('Ycenter'):.2f}" if centre.get("Ycenter") is not None else None,
            }
            formatted_history.append({
                "ID": entry.get("ID"),
                "Time": time_str,
                "Object Name": entry.get("Object Name"),
                "Bounding Box": bb_json,
                "Center Point": centre_json,
                "Depth": f"{entry.get('Depth'):.2f}" if entry.get("Depth") is not None else None,
                "Confidence Level": f"{entry.get('Confidence Level'):.2f}" if entry.get("Confidence Level") is not None else None,
            })
        # Write the history to the JSON file (overwriting each time)
        if self.jsonl_enabled and self.jsonl_path:
            try:
                with open(self.jsonl_path, "w", encoding="utf-8") as f:
                    json.dump(formatted_history, f, indent=2)
            except Exception as e:
                self.get_logger().warn(f"Failed to write detection history to JSON: {e}")
        # Update the global history for the HTTP API
        global recent_detection_history  # type: ignore[declaration-of-linked-name]
        recent_detection_history = formatted_history
        # Update the global latest detection data for compatibility
        if formatted_history:
            latest = formatted_history[-1]
            latest_detection_data["timestamp"] = now
            latest_detection_data["detector"] = self.detector_name
            latest_detection_data["fps"] = self.fps
            latest_detection_data["detections"] = [
                {
                    "Object Name": latest.get("Object Name"),
                    "Bounding Box": latest.get("Bounding Box"),
                    "Center Point": latest.get("Center Point"),
                    "Depth": latest.get("Depth"),
                    "Confidence Level": latest.get("Confidence Level"),
                }
            ]
        # Reset the last update time
        self.last_json_update_time = now
    # ------------------------------------------------------------------
    # Clean up on destruction
    def destroy_node(self) -> None:  # type: ignore[override]
        self.get_logger().info("Shutting down Visual Perception Node")
        try:
            if self.jsonl_file:
                self.jsonl_file.close()
        except Exception:
            pass
        super().destroy_node()


def main() -> None:
    """Entry point for the script."""
    # If rclpy isn't available, inform user and exit
    if rclpy is None:
        print("This script requires rclpy (ROS2 Python) to run. Please ensure you are in a ROS2 environment.")
        return
    rclpy.init()
    node = VisualPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
