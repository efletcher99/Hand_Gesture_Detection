import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Hands setup.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
MARGIN = 20  # Margin to add around the bounding box


def draw_bounding_box(image, landmarks, margin=MARGIN):
    h, w, _ = image.shape
    landmark_coords = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]

    x_min = max(min([pt[0] for pt in landmark_coords]) - margin, 0)
    y_min = max(min([pt[1] for pt in landmark_coords]) - margin, 0)
    x_max = min(max([pt[0] for pt in landmark_coords]) + margin, w)
    y_max = min(max([pt[1] for pt in landmark_coords]) + margin, h)

    return (x_min, y_min, x_max, y_max)


def crop_to_bounding_box(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max], (x_max - x_min, y_max - y_min)


def adjust_landmarks_to_bbox(landmarks, bbox, bbox_size):
    x_min, y_min, x_max, y_max = bbox
    bbox_width, bbox_height = bbox_size
    adjusted_landmarks = []

    for landmark in landmarks.landmark:
        x_rel = (landmark.x * img.shape[1] - x_min) / bbox_width
        y_rel = (landmark.y * img.shape[0] - y_min) / bbox_height
        adjusted_landmarks.append((x_rel, y_rel))

    return adjusted_landmarks


for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks.
                mp_drawing.draw_landmarks(
                    img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box with margin.
                bbox = draw_bounding_box(img_rgb, hand_landmarks)
                print(f"Bounding box for {img_path}: {bbox}")
                cropped_img, bbox_size = crop_to_bounding_box(img_rgb, bbox)

                if cropped_img.size == 0:
                    print(f"Warning: Cropped image is empty for {img_path}. Skipping.")
                    continue

                # Adjust landmarks to be relative to the bounding box.
                adjusted_landmarks = adjust_landmarks_to_bbox(hand_landmarks, bbox, bbox_size)

                # Create a blank image for displaying adjusted landmarks.
                hand_only_img = cropped_img.copy()
                h, w, _ = hand_only_img.shape

                for x_rel, y_rel in adjusted_landmarks:
                    cv2.circle(hand_only_img, (int(x_rel * w), int(y_rel * h)), 5, (0, 0, 128), -1)

                plt.figure()
                plt.imshow(cv2.cvtColor(hand_only_img, cv2.COLOR_RGB2BGR))
                plt.title(f"Class: {dir_}")

plt.show()