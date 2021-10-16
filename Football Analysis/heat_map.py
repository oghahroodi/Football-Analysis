import numpy as np
import cv2
import copy

video_path = 'videos/1.mp4'


def get_heat_map():
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    num_frames = 3500

    first_iteration_indicator = 1
    for i in range(0, num_frames):

        if (first_iteration_indicator == 1):
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fgmask = fgbg.apply(gray)

            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(
                fgmask, thresh, maxValue, cv2.THRESH_BINARY)

            accum_image = cv2.add(accum_image, th1)

            # cv2.imshow('q', fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # cv2.imwrite('diff-overlay.jpg', result_overlay)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_heat_map()
