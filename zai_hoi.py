import os
import base64
from pathlib import Path
import ast
import cv2

import zai
from zai import ZaiClient

# ---------------------------------------
#  Get Zai API key and Intantiate Client
# ---------------------------------------

ZAI_API_KEY = os.getenv('ZAI_API_KEY')

client = ZaiClient(api_key=ZAI_API_KEY)

# ---------------------------------------
#  GLM VLM Call to glm-4.6v-flash
#  Returns coordinates of bounding boxes
#  and the interation vector
# ---------------------------------------

def image_hoi_detect(image_url, image_dim, vlm_model = "glm-4.6v-flash"):
    try:
        response = client.chat.completions.create(
            model = vlm_model,
            messages = [
                {
                    "content": 
                        """
                        You are an expert in human-object-interaction detection and particularly throwing HOI detection.

                        You are given an image and its dimension, most likely with a human and an object they are interacting with or throwing, detect the interaction and give coordinates of bounding boxes and interaction vector. The format is specified below.

                        RULES:
                        - Prioritze throwing actions.
                        - Follow the given format ABSOLUTELY.
                        - NO leading or following text or output.
                        - Give the response in JSON format.
                        - DO NOT include newlines in the content of the response's message.
                        - YOU MUST make the coordinates fit within the given dimension.

                        Output Format:
                        {{"human_bbox": (xmin, ymin, xmax, ymax), "object_bbox": (xmin, ymin, xmax, ymax), "interaction_vector": (human, verb, object)}}
                        """,
                    "role": "system"
                },
                {
                    "content" : [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "The dimension of the image in representation (width, height) is: {image_dim}"
                        },
                        {
                            "type": "text",
                            "text": "What is the human-object-interaction in this image of dimension {image_dim}, if there is one. Provide bounding box coordinates of the person and object they're interacting with in (xmin, ymin, xmax, ymax) format. Be sure to have the bounding box coordinates fit within the image dimension. Additionally, provide the interaction vector in the format (human, verb, object)."
                        }
                    ],
                    "role": "user"
                }
            ],
            thinking = {
                "type": "enabled"
            },
            stream = False
        )
    except Exception as e:
        print(f"The call to GLM model failed: {e}")
        return e

    return response

# ----------------------------------------------
#  Function to Convert the Local Path to Base64
# ----------------------------------------------

def local_image_to_base64_url(file_path):
    if file_path.endswith(".png"):
        mime_type = "image/png"
    elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
        mime_type = "image/jpeg"
    else:
        raise ValueError("unsupport image format")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base65,{encoded_string}"

# -----------------------------------
#  Go Retrieve Frames from Directory
# -----------------------------------

video_dirs = Path(__file__).parent / "testing_frames"

# videos dict where the frame's path url will be stored
video_frames = {}

for vid_dir in video_dirs.iterdir():
    frames = [(local_image_to_base64_url(str(frame)), frame) for frame in vid_dir.iterdir() if frame.is_file()]
    video_frames[vid_dir.name] = frames

# -----------------------------------
#  Call the VLM for the Frames
# -----------------------------------

for vid_dir, frames in video_frames.items():
    os.makedirs(Path(__file__).parent / 'classified_and_detected_frames' / f'{vid_dir}', exist_ok=True)

    # call the hoi detection on every frame of a video (that was captured)
    for frame in frames:
        # read the image into OpenCV
        image = cv2.imread(frame[1])

        # get height and width of image
        height, width, *_ = image.shape

        image_dim = (width, height)

        # get responses
        response = image_hoi_detect(frame[0], image_dim)

        try:
            response_content = ast.literal_eval(response.choices[0].message.content)
        except Exception as e:
            print(f"Failed to cast response as dictionary/json: {e}")
            continue

        # ---------------------------------
        #  Get Coordinates, and Normalize
        # ---------------------------------

        human_top_left = (round(response_content['human_bbox'][0] / 1000 * width), round(response_content['human_bbox'][1] / 1000 * height))
        human_bottom_right = (round(response_content['human_bbox'][2] / 1000 * width), round(response_content['human_bbox'][3] / 1000 * height))

        object_top_left = (round(response_content['object_bbox'][0] / 1000 * width), round(response_content['object_bbox'][1] / 1000 * height))
        object_bottom_right = (round(response_content['object_bbox'][2] / 1000 * width), round(response_content['object_bbox'][3] / 1000 * height))

        # get the interaction vector
        interaction_vector = response_content['interaction_vector']

        if len(interaction_vector) != 3:
            print(f"Interaction vector not sufficient")
            continue

        # -----------------------------------
        #  Creating the Bounding Boxes
        # -----------------------------------

        # define color and thickness of bbox
        humanbbox_color = (255, 0, 0)
        objectbbox_color = (0, 0, 255)
        thickness = 3

        # draw rectangles
        # human bbox
        cv2.rectangle(image, human_top_left, human_bottom_right, humanbbox_color, thickness)

        # put the label for the Human bounding box
        human_bbox_label = interaction_vector[0]
        label_color = (48, 189, 45)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        text_org = (human_top_left[0], human_top_left[1] - 15)
        image = cv2.putText(image, human_bbox_label, text_org, font, font_scale, label_color, font_thickness, cv2.LINE_AA)

        # object bbox
        cv2.rectangle(image, object_top_left, object_bottom_right, objectbbox_color, thickness)

        # put the label for the object bounding box
        object_bbox_label = interaction_vector[2]

        text_org = (object_top_left[0], object_top_left[1] - 15)
        image = cv2.putText(image, object_bbox_label, text_org, font, font_scale, label_color, font_thickness, cv2.LINE_AA)

        # put the interaction vector above the human's bounding box
        interaction_vector_label = str(interaction_vector)
        text_org = (human_top_left[0], human_bottom_right[1] - 15)
        image = cv2.putText(image, interaction_vector_label, text_org, font, font_scale, label_color, font_thickness, cv2.LINE_AA)

        # ------------------------------------
        #  Show and Save the Image
        # ------------------------------------

        '''
        # show from terminal
        cv2.imshow(f"{interaction_vector}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        # save to folder
        try:
            cv2.imwrite(Path(__file__).parent / 'classified_and_detected_frames' / f'{vid_dir}' / f'{frame[1].name}', image)

            # if sccessful delete the processed image from the testing_frames folder
            os.remove(frame[1])

            # and remove from the frames list
            frames.remove(frame)
        except Exception as e:
            print(f"Failed to write image with bounding box to folder: {e}")
