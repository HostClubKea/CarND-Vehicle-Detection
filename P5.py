from src.svc_classifier import SvcClassifier
from src.windows_slider import WindowsSlider
from src.vehicle_tracker import VehicleTracker
from src.cnn_classifier import CnnClassifier
from src.utils import *

if __name__ == '__main__':
    # Create SVN classifier
    svc_classifier = SvcClassifier()
    # Create windows slider
    windows_slider = WindowsSlider()

    cnn_classifier = CnnClassifier()
    # Create vehicle tracker
    vehicle_tracker = VehicleTracker(windows_slider, classifiers=[svc_classifier], continuous=False)

    images, names = get_images('test_images')

    import time
    t1 = time.time()

    for i in range(len(images)):
        print("Image" + str(i))
        image = vehicle_tracker.process(images[i])

        mpimg.imsave("output_images/{}.png".format(names[i]), image)

    from moviepy.editor import VideoFileClip

    vehicle_tracker = VehicleTracker(windows_slider, classifiers=[svc_classifier])

    # video_output_name = 'test_video_vehicles.mp4'
    # video = VideoFileClip("test_video.mp4")
    # video_output = video.fl_image(vehicle_tracker.process)
    # video_output.write_videofile(video_output_name, audio=False)

    # video_output_name = 'project_video_vehicles.mp4'
    # video = VideoFileClip("project_video.mp4")
    # video_output = video.fl_image(vehicle_tracker.process)
    # video_output.write_videofile(video_output_name, audio=False)

    elapsed_time1 = time.time() - t1
    print("\r\nTotal - " + str(elapsed_time1))



