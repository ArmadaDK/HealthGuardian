import mediapipe as mp
print(dir(mp))
# 看看输出里有没有 'solutions'
if 'solutions' in dir(mp):
    print(dir(mp.solutions))
    # 看看输出里有没有 'face_mesh'