import pickle

import cv2
import mediapipe as mp
import numpy as np

last_index = 0
letter_count = 0
letters_predicted = []
current_prediction_index = 0

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

word = ""

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h',
               'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'n': 'n', 't': 't'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[prediction[0]]

        letters_predicted.append(predicted_character)
        if letters_predicted[current_prediction_index] == predicted_character:
            letter_count += 1
        else:
            letter_count = 0

        current_prediction_index += 1

        if letter_count > 15:
            word += predicted_character
            letter_count = 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.3
        thickness = 3
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        text_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
        text_x = (W - text_size[0]) // 2
        text_y = H - 20  # Position 20 pixels from the bottom

        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10), bg_color, cv2.FILLED)
        cv2.putText(frame, word, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()