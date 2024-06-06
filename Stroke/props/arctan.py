#arctanを計算
import math

# x = 1
# y = 1
z = 1

#視野角 H*V
#Kinectのdepth視野角 75° x 65°, 120° x 120°
#KinectのRGB視野角 90° x 59°, 90° x 74.3°
#RealSenseのdepth視野角 85.2°x 58°
#RealSenseのRGB視野角 69.4° x 42.5°

FOV_H = 42.5
theta = math.radians(FOV_H/2)
y = 1 / math.tan(theta)
print(f"y = {y}")

