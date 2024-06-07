import os
import cv2

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 500
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

cap = cv2.VideoCapture(1)  # Changed to 0 for default camera

for j in range(len(letters)):
    class_dir = os.path.join(DATA_DIR, str(letters[j]))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(letters[j]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'Q', collecting data  for class {}".format(letters[j]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Small delay

        # Save the captured image to the correct folder
        image_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(image_path, frame)

        counter += 1

    print(f"Finished collecting data for class {letters[j]}")

cap.release()
cv2.destroyAllWindows()
