import math

def calculate_angle(vector1, vector2):
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle)

# ベクトルの値を指定してなす角を計算する例
vector1 = [13.6,-28.6]
vector2 = [13.6,-35.7]
angle = calculate_angle(vector1, vector2)
print(angle)
