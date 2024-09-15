from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import numpy as np
import cv2

model = YOLO("yolov8n.pt")
source_video = "driving.mp4"
target_video = "distance.mp4"
confidence_threshold = 0.3
IOU_threshold = 0.5

SOURCE = np.array([
    [-800, 480],
    [960, 480],
    [404, 304],
    [317, 305]
])

TARGET_WIDTH = 30
TARGET_HEIGHT = 100

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

video_info = sv.VideoInfo.from_video_path(source_video)
frame_generator = sv.get_video_frames_generator(source_video)

byte_track = sv.ByteTrack(track_thresh=confidence_threshold, frame_rate=video_info.fps)

thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
trace_annotator = sv.TraceAnnotator(thickness=thickness, 
                                    trace_length=video_info.fps * 2, 
                                    position=sv.Position.BOTTOM_CENTER)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, 
                                    text_thickness=thickness, 
                                    text_position=sv.Position.BOTTOM_CENTER)

with sv.VideoSink(target_video, video_info) as sink:

    # loop over source video frame
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections[detections.class_id != 0]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_threshold)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # format labels
        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            labels.append(f"#{tracker_id} {y}m")

        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # add frame to target video
        sink.write_frame(annotated_frame)