# file to run the camera and collect the frames, 
# show the frames in realtime through window, 
# and save in low frame rate to folder named frames

# imports
import os
import cv2

# make the directory for all captured testing videos, if it does not exist
os.makedirs("./testing_frames", exist_ok=True)

# get the current length (sub directories of testing_frames,
# to get the current amount of vids captures, so I can make a directory for the new one
new_folder_id = len([folder for folder in os.listdir('./testing_frames')]) + 1

# then create the new sub directory
new_folder_path = f"./testing_frames/frame_folder_vid_{new_folder_id}"
os.makedirs(new_folder_path, exist_ok=True)

# the function to extract the frames from live camera with skipping frames
def extract_frames(max_frames, skip_frames=0, start_frame_count=0):
    # open the connected camera (Logi C270)
    frame_cap = cv2.VideoCapture(1)

    # set resolution (using lesser resolution for storage efficiency)
    frame_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
    frame_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # check if the camera is opened
    if not frame_cap.isOpened():
        raise RuntimeError("Could not open webcame. Try changing the device index")

    curr_frame_count = start_frame_count 
    while True:
        ret, frame = frame_cap.read()

        # check if the camera is reading/returning or if the current frames are at their maximum
        if not ret or curr_frame_count > start_frame_count + max_frames:
            break

        curr_frame_count += 1

        # skip the frames
        for _ in range(skip_frames):
            ret, _ = frame_cap.read()

            # still check if there is no return
            if not ret:
                break

        # write to folder with filename
        frame_filename = os.path.join(new_folder_path, f"{curr_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    # release and destroy windows
    frame_cap.release()
    cv2.destroyAllWindows()

# call the function to extract the frames
extract_frames(2000, 15)
