# Iris Detection

import sensor, time, image

# Reset sensor
sensor.reset()

# Sensor settings
sensor.set_contrast(3)
sensor.set_gainceiling(16)

# Set resolution to VGA.
sensor.set_framesize(sensor.VGA)

# Crop image to 200x100, which gives more details with less data to process
sensor.set_windowing((300, 250, 300, 200))

sensor.set_pixformat(sensor.GRAYSCALE)

# Load Haar Cascade
# By default this will use all stages, lower satges is faster but less accurate.
eyes_cascade = image.HaarCascade("eye", stages=500)
print(eyes_cascade)

# FPS clock
clock = time.clock()

while (True):
    clock.tick()
    # Capture snapshot
    img = sensor.snapshot()
    # Find eyes !
    # Note: Lower scale factor scales-down the image more and detects smaller objects.
    # Higher threshold results in a higher detection rate, with more false positives.
    eyes = img.find_features(eyes_cascade, threshold=1.5, scale_factor=1.5)

    # Find iris
    for e in eyes:
        iris = img.find_eye(e)
        img.draw_rectangle(e)
        img.draw_cross(iris[0], iris[1])

    # Print FPS.
    # Note: Actual FPS is higher, streaming the FB makes it slower.
    print(clock.fps())
