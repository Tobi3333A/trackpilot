class Metrics:
    def __init__(self):
        self.pipeline_times = []
        self.track_times = []

        self.tracked_frames = 0
        self.lost_frames = 0

        self.target_ids_seen = set()

    def record_pipeline_time(self, elapsed_time):
        self.pipeline_times.append(elapsed_time)

    def record_track_time(self, elapsed_time):
        self.track_times.append(elapsed_time)

    def record_tracking_state(self, state):
        if state == 'Tracking':
            self.tracked_frames += 1
        elif state == "Target Lost":
            self.lost_frames += 1

    def record_target_id(self, target_id):
        self.target_ids_seen.add(target_id)

    def print_summary(self):
        total_frames = len(self.pipeline_times)

        print(f"Total frames: {total_frames}")

        if self.pipeline_times:
            total_time = sum(self.pipeline_times)
            avg_frame_fps = total_frames / total_time
            print(f"Average Frame FPS: {avg_frame_fps:.2f}")

        if self.track_times:
            total_time = sum(self.track_times)
            avg_track_fps = total_frames / total_time
            print(f"Average Track FPS: {avg_track_fps:.2f}")

        frames = self.tracked_frames + self.lost_frames

        if frames > 0:
            success_rate = (self.tracked_frames / frames) * 100
            print(f"Tracking success rate: {success_rate:.2f}%")

        print(f"Target IDs seen: {sorted(self.target_ids_seen)}")
        print(f"Unique Target IDs: {len(self.target_ids_seen)}")
