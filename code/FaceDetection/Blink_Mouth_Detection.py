import cv2
import dlib
import numpy as np
from imutils import face_utils


def _help():
    print("Usage:")
    print("     python liveness_detect.py")
    print("     python liveness_detect.py <path of a video>")
    print("For example:")
    print("     python liveness_detect.py video/lee.mp4")
    print("If the path of a video is not provided, the camera will be used as the input.Press q to quit.")


def eye_aspect_ratio(eye):
    # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    # (|m2-m9|+|m4-m7|)/(2|m0-m6|)
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


def liveness_detection(vs, file_stream):
    EAR_THRESH = 0.2
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 2
    MAR_THRESH = 0.6
    frame_num = 0
    # 初始化眨眼的连续帧数以及总的眨眼次数
    blink_counter = [0, 0]  # left eye and right eye
    blink_total = [0, 0]  # left eye and right eye

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    print("[INFO] starting video stream thread...")
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if file_stream and not vs.more():
            break

        frame = vs.read()
        if frame is not None:
            frame_num += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            # 只能处理一张人脸
            if len(rects) == 1:
                rect = rects[0]
                shape = predictor(gray, rect)  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                if left_ear < EAR_THRESH:
                    blink_counter[0] += 1

                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter[0] and blink_counter[0] <= EAR_CONSEC_FRAMES_MAX:
                        blink_total[0] += 1

                    blink_counter[0] = 0

                cv2.putText(frame, "LBlinks: {}".format(blink_total[0]), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "LEAR: {:.2f}".format(left_ear), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if right_ear < EAR_THRESH:
                    blink_counter[1] += 1

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter[1] and blink_counter[1] <= EAR_CONSEC_FRAMES_MAX:
                        blink_total[1] += 1

                    blink_counter[1] = 0

                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                cv2.putText(frame, "RBlinks: {}".format(blink_total[1]), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "REAR: {:.2f}".format(right_ear), (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Mouth: {}".format("open" if mar > MAR_THRESH else "closed"),
                            (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (450, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(rects) == 0:
                cv2.putText(frame, "No face!", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "More than one face!", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if blink_total[0] > 2 and blink_total[1] > 2 :
              if mar > MAR_THRESH:
                cv2.putText(frame, "Thank you for your cooperation!", (130, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
                cv2.putText(frame, "Now you can press q to quit.", (130, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
                cv2.putText(frame, "A photo from the video will be saved for identification purposes.", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
            else:
                cv2.putText(frame, "If you are wearing glasses, please take them off.", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
                cv2.putText(frame, "Please blink more then 3 times ...", (130, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
                cv2.putText(frame, "And open your mouth ...", (130, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)

            # 按下q键退出循环（鼠标要点击一下图片使图片获得焦点）
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_num > 500:
                cv2.imwrite('examples/image/' + str(frame_num) + '.jpg', frame)
                break

    cv2.destroyAllWindows()
    vs.stop()
    if blink_total[0] > 2 and blink_total[1] > 2 :
     if mar > MAR_THRESH:
        return 1
     else:
        return 0
    else:
        return 0
