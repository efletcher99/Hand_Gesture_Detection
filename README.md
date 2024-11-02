# Hand Gesture Detection
**Hand Gesture Detection** is an interactive machine learning project that enables users to create a custom hand gesture dataset for alphabet letters, 
including a user-defined "space" gesture. Users can then build, train, and test a model based on their recorded gestures to recognize hand-signed letters.

## Features
- Customizable Dataset Creation: Record unique hand gestures for each letter in the alphabet, including a "space" gesture, to build a personalized dataset.
- Model Training: Build and train a machine learning model using your custom dataset.
- Gesture Recognition Testing: Test the model’s accuracy in recognizing real-time hand gestures.

## Development Setup
1. Create virtual environment 'python3 -m venv .venv'
2. Activate virual environment 'source .venv/bin/activate'
3. Intstall necessary libraries 'pip install -r requirements.txt'

## Getting Started
1. Record Gestures: Run 'main.py' to record hand gestures for each letter.
2. Create the Dataset: Run 'create_dataset.py' to organize the recorded hand gestures into a dataset.
3. Display the Dataset: Run 'display_dataset.py' to view each letter in the dataset.
4. Build the Model: Run 'train_model.py' to train a machine learning model using your custom dataset.
5. Test the Model: Run 'test_model.py' to sign letters and observe as they are recognized and displayed on the screen.
