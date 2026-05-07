import cv2
import mediapipe as mp
import time
import winsound
import threading
from queue import Queue
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- 新增：异步报警线程 ---
class AlarmWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.daemon = True  # 守护线程：主程序退出时自动关闭
        self.start()

    def run(self):
        while True:
            # 获取报警类型：(频率, 时长)
            alarm_type = self.queue.get()
            if alarm_type is None: break
            frequency, duration = alarm_type
            winsound.Beep(frequency, duration)
            self.queue.task_done()

    def trigger(self, freq, duration):
        # 如果队列堆积太多，就不再添加，防止报警延迟太久
        if self.queue.qsize() < 2:
            self.queue.put((freq, duration))


# --- 修改后的主类 ---
class ModernHealthGuardian:
    def __init__(self, model_path='face_landmarker_v2_with_blendshapes.task'):
        # 初始化报警器
        self.alarm = AlarmWorker()

        # MediaPipe 初始化 (保持不变)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.last_blink_time = time.time()
        self.baseline_dist = 0
        self.calibrated = False

    def get_ear(self, landmarks, indices):
        p = [landmarks[i] for i in indices]
        v1 = ((p[1].x - p[5].x) ** 2 + (p[1].y - p[5].y) ** 2) ** 0.5
        v2 = ((p[2].x - p[4].x) ** 2 + (p[2].y - p[4].y) ** 2) ** 0.5
        h = ((p[0].x - p[3].x) ** 2 + (p[0].y - p[3].y) ** 2) ** 0.5
        return (v1 + v2) / (2.0 * h)

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape

                # 1. 眨眼逻辑
                ear = (self.get_ear(landmarks, self.LEFT_EYE) + self.get_ear(landmarks, self.RIGHT_EYE)) / 2.0
                if ear < 0.2:
                    self.last_blink_time = time.time()

                # 2. 距离逻辑
                p1, p2 = landmarks[468], landmarks[473]
                dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

                if not self.calibrated:
                    self.baseline_dist += dist
                    frame_count += 1
                    cv2.putText(frame, f"Calibrating: {frame_count}/50", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    if frame_count >= 50:
                        self.baseline_dist /= 50
                        self.calibrated = True
                else:
                    # --- 重点：使用 self.alarm.trigger 替代 winsound.Beep ---
                    # 前倾警告
                    if dist > self.baseline_dist * 1.25:
                        cv2.putText(frame, "TOO CLOSE!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.alarm.trigger(400, 200)

                        # 疲劳警告
                    if time.time() - self.last_blink_time > 8:
                        cv2.putText(frame, "BLINK NOW!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.alarm.trigger(800, 100)

                # UI 绘制（略）
                color = (0, 255, 0) if ear > 0.2 else (0, 0, 255)
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Modern Guardian', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 记得把下载的 .task 文件放在同目录下
    guardian = ModernHealthGuardian('face_landmarker_v2_with_blendshapes.task')
    guardian.run()