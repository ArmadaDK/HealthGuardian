# import mediapipe as mp
# print(dir(mp))
# # 看看输出里有没有 'solutions'
# if 'solutions' in dir(mp):
#     print(dir(mp.solutions))
#     # 看看输出里有没有 'face_mesh'

import winsound
# 尝试播放一个 1000Hz 持续 500ms 的声音
print("准备发声...")
winsound.Beep(1000, 500)
print("发声结束")