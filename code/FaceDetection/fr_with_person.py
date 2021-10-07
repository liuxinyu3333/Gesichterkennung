import os
import blink_detection as bd
import sys
import face_recognition_knn as frk
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import glob

def _help():
    print("Usage:")
    print("     python fr_with_person.py")
    print("     python fr_with_person.py <path of a video>")
    print("For example:")
    print("     python fr_with_person.py video/lee.mp4")
    print("If the path of a video is not provided, the camera will be used as the input.Press q to quit.")


num = 0

if len(sys.argv) > 2 or "-h" in sys.argv or "--help" in sys.argv:
    _help()
elif len(sys.argv) == 2:
    vs = FileVideoStream(sys.argv[1]).start()
    file_stream = True
    num = bd.blink_detection(vs, file_stream)
else:
    vs = VideoStream(src=0).start()
    file_stream = False
    num = bd.blink_detection(vs, file_stream)

if num == 1:
    frk.knn_face_recognition(model_path="face_recog_model.clf",test_path="examples/image")
else:
    print("This is not a real person")


path = "examples/image"
paths = glob.glob(os.path.join(path, '*.jpg'))

for file in paths:
    os.remove(file)