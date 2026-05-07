import cv2
import mediapipe as mp
import time
import winsound
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ModernHealthGuardian:
    def __init__(self, model_path='face_landmarker_v2_with_blendshapes.task'):
        # 1. 配置新版 Task 参数
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO  # 视频流模式
        )
        # 2. 创建检测器
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 关键点索引 (新旧版本索引一致)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self.last_blink_time = time.time()
        self.baseline_dist = 0
        self.calibrated = False

    def get_ear(self, landmarks, indices):
        # 新版 landmarks 是对象列表，通过 .x, .y 访问
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

            # 转换格式：OpenCV (BGR) -> MediaPipe (RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 必须传入毫秒级时间戳
            timestamp_ms = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape  # 获取图像宽高用于坐标转换

                # 眨眼逻辑
                ear = (self.get_ear(landmarks, self.LEFT_EYE) + self.get_ear(landmarks, self.RIGHT_EYE)) / 2.0
                # 在屏幕上实时显示 EAR 值（眼睛张开度）
                color = (0, 255, 0) if ear > 0.2 else (0, 0, 255)  # 闭眼变红
                cv2.putText(frame, f"Eye Open: {ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if ear < 0.2: self.last_blink_time = time.time()

                # 前倾逻辑 (瞳孔距离)
                p1, p2 = landmarks[468], landmarks[473]
                cv2.circle(frame, (int(p1.x * w), int(p1.y * h)), 4, (255, 0, 0), -1)
                cv2.circle(frame, (int(p2.x * w), int(p2.y * h)), 4, (255, 0, 0), -1)
                dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

                if self.calibrated:
                    # 显示当前距离比例
                    ratio = dist / self.baseline_dist
                    dist_color = (0, 255, 0) if ratio < 1.2 else (0, 0, 255)
                    cv2.putText(frame, f"Forward Ratio: {ratio:.2f}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, dist_color, 2)

                if not self.calibrated:
                    self.baseline_dist += dist
                    frame_count += 1
                    if frame_count >= 50:
                        self.baseline_dist /= 50
                        self.calibrated = True
                        print("新版模型标定完成！")
                else:
                    if dist > self.baseline_dist * 1.25:
                        winsound.Beep(400, 200)  # 前倾警告
                    if time.time() - self.last_blink_time > 8:
                        winsound.Beep(800, 100)  # 眨眼警告

            cv2.imshow('Modern Guardian', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 记得把下载的 .task 文件放在同目录下
    guardian = ModernHealthGuardian('face_landmarker_v2_with_blendshapes.task')
    guardian.run()