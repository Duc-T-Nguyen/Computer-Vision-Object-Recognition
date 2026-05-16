import cv2
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import time

modelid = "Qwen/Qwen2.5-VL-3B-Instruct"
revision = "main"
prompt = (
    "Describe the primary action in this image using exactly three words: Subject, Action, and Object. "
    "Output strictly as a JSON array of strings. Do not use articles like 'a' or 'the'. "
    "Do not include any other text, markdown, or punctuation outside the array.\n\n"
    "Examples:\n"
    "[\"person\", \"holding\", \"ball\"]\n"
    "[\"cat\", \"eating\", \"mouse\"]\n"
    "[\"dog\", \"chasing\", \"stick\"]\n\n"
    "Now describe this image:"
)

def model_setup():
    my_model_path = "F:\myScriptModels"
    print(f"Loading {modelid} with revision {revision} onto GPU")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(modelid, revision=revision, cache_dir=my_model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        modelid, 
        revision=revision, 
        quantization_config=bnb_config, 
        device_map={"":"cuda"}, 
        cache_dir=my_model_path
    )
    
    print("model loaded.")
    return model, processor

def extract_hoi(response):
    try:
        clean_text = response.strip().strip('`').replace('json', '', 1).strip()
        hoi_response = json.loads(clean_text)
        if isinstance(hoi_response, list) and len(hoi_response) == 3:
            return hoi_response
        else:
            return ['error', 'invalid format', str(hoi_response)]
    except json.JSONDecodeError:
        return ['error', 'parsing failed', response]

model, processor = model_setup()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()
print("Starting HOI protocol. Press Q to quit.")

while True:
    cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    start_time = time.time()

    messages = [
        {
            "role":"user",
            "content": [
                {"type": "image", "image":pil_image},
                {"type": "text", "text":prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=15)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time

    hoi_vector = extract_hoi(answer)
    print(f"FPS: {fps:.2f} | HOI Vector: {hoi_vector}")

    cv2.putText(frame, str(hoi_vector), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Jetson VLM-based HOI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()