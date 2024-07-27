Driver Drowsiness Detection System
Overview
The Driver Drowsiness Detection System is designed to monitor and detect signs of drowsiness in drivers using facial landmark detection. The system employs the Dlib library and a pre-trained shape predictor model to analyze facial expressions and eye movements. When signs of drowsiness are detected, the system activates an alarm to alert the driver, ensuring a safer driving experience.

Features
Real-time drowsiness detection using facial landmarks
Integration with hardware to activate an alarm
Easy setup and deployment
Requirements
Python 3.x
Dlib library
OpenCV
NumPy
Pre-trained shape predictor model (shape_predictor_68_face_landmarks.dat)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Download the pre-trained shape predictor model:

You can download the shape_predictor_68_face_landmarks.dat file from the Dlib website or directly from this link. Extract the file and place it in the models directory of the project.

Ensure you have the necessary hardware components for the alarm system (e.g., buzzer, Arduino) and connect them as per the provided instructions.

Usage
Run the detection script:

bash
Copy code
python detect_drowsiness.py
The system will start capturing video from the default webcam. If drowsiness is detected, the alarm will be activated.

Directory Structure
css
Copy code
driver-drowsiness-detection/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat
│
├── src/
│   ├── detect_drowsiness.py
│   ├── alarm_system.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
The Dlib library for providing the pre-trained shape predictor model.
OpenCV and NumPy for their powerful computer vision and numerical processing capabilities.
Contact
For any questions or feedback, please open an issue on the GitHub repository or contact the maintainer at your-
kiransingh131211@gmail.com
