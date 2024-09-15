import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO('yolov8n.pt')
source_video = 'vehicles.mp4'
target_video = 'count.mp4'

confidence_threshold = 0.3
IOU_threshold = 0.5

video_info = sv.VideoInfo.from_video_path(source_video)
frame_generator = sv.get_video_frames_generator(source_video)

thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
byte_track = sv.ByteTrack(track_thresh=confidence_threshold, frame_rate=video_info.fps)

start, end = sv.Point(x=0, y=1080), sv.Point(x=3840, y=1080)
line_zone = sv.LineZone(start=start, end=end)
line_annotator = sv.LineZoneAnnotator(thickness=thickness)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, 
                                    text_thickness=thickness, 
                                    text_position=sv.Position.BOTTOM_CENTER)

with sv.VideoSink(target_video, video_info) as sink:

    # loop over source video frame
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)
        crossed_in, crossed_out = line_zone.trigger(detections)

        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = line_annotator.annotate(
            frame=annotated_frame, line_counter=line_zone
        ) 
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        sink.write_frame(annotated_frame)