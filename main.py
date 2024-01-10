import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN, extract_face
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=2
).to(device)
resnet.load_state_dict(torch.load('model.pt'))
resnet.eval()

mtcnn = MTCNN(keep_all=True,
                  image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)

font = ImageFont.truetype('Arial.ttf', 20)

persons = ['Idoyatov Zufar', 'Zhashkey Yerkanat']

photos = [
    'test_images_orig/idoyatov_zufar/photo_5192662130434887241_y.jpg',
    'test_images_orig/zhashkey_yerkanat/photo_5192662130434887289_y.jpg'
]

photos_opened = [
    Image.open(photos[0]),
    Image.open(photos[1])
]

aligned = [
    mtcnn(photos_opened[0]),
    mtcnn(photos_opened[1])
]

aligned = torch.stack(aligned).to(device).reshape(2, 3, 160, 160)
embeddings = resnet(aligned).detach().cpu()
print(embeddings)

def distance_to_corpus(single_embedding, corpus_embeddings):
    """
    Computes the distances between a single embedding and a corpus of embeddings.

    Parameters:
    - single_embedding: The embedding of the single photo.
    - corpus_embeddings: The embeddings of the corpus.

    Returns:
    - distances: List of distances between the single embedding and each corpus embedding.
    """
    distances = [(single_embedding - e2).norm().item() for e2 in corpus_embeddings]
    return np.array(distances)

def get_new_image(img):
    img = Image.fromarray(img).rotate(90)
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    x_aligned = mtcnn(img)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    
    if boxes is not None:
        for i, (box, point) in enumerate(zip(boxes, points)):
            draw.rectangle(box.tolist(), width=5)
            for p in point:
                draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
            if len(x_aligned.shape) < 4: 
                embedding = resnet(x_aligned.expand((1,-1,-1,-1))).detach().cpu()
            else:
                embedding = resnet(x_aligned[i].expand((1,-1,-1,-1))).detach().cpu()
            distances = distance_to_corpus(embedding, embeddings)
            found_person = persons[distances.argmin()]
            conf_score = distances.min()
            if conf_score > 3: 
                found_person = 'Unknow Person'
            result_text = f"{found_person} (distance: {str(np.round(conf_score, 3))}), detection score: {str(np.round(probs[i], 3))})"
            draw.text((box[0], box[3]+10), result_text, stroke_width=2, stroke_fill='black', font=font, fill='#37eb34' if conf_score <= 3 else '#fc3503')
    return img_draw

rtsp_url = 'rtsp://admin:admin@172.20.10.2:1935'
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

frame_rate = 2
prev = 0

while True:
    # Read a frame from the video stream
    time_elapsed = time.time() - prev
    start = time.time()
    ret, frame = cap.read()
    
    if time_elapsed > 1./frame_rate and frame is not None:
        prev = time.time()
    
        new_frame = get_new_image(frame)
        new_frame_arr = np.array(new_frame)
        cv2.imshow('Face Recognition', new_frame_arr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()