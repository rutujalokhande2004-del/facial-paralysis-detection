# Facial Expression Analysis Project

This project analyzes facial expressions in real-time using computer vision and generates detailed reports of facial muscle movements.

## Features

- Real-time facial expression detection
- Analysis of multiple facial muscle movements
- PDF report generation
- JSON data export
- Visual feedback through OpenCV interface

## Prerequisites

- Python 3.8 or higher
- Webcam
- Windows OS (for text-to-speech functionality)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rutujalokhande2004-del/facial-paralysis-detection.git
   ```

2. change the directory

   ```bash
   cd facial-paralysis-detection
   ```

3. Create a virtual environment (recommended):

   ```bash
   python3.10 -m venv venv
   ```

4. activate the virtual environemnt

    ```bash
    venv\scripts\activate
    ```

5. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `index.py`: Entry point of the application
- `requirements.txt`: List of Python dependencies

## Usage

1. Activate your virtual environment (if using one):

   ```bash
   venv\scripts\activate
   ```

2. Run the application:

   ```bash
   python index.py
   ```

3. Position yourself in front of the webcam
4. The program will analyze your facial expressions in real-time
5. When finished, press 'q' to quit and generate the reports

## Features Detected

The system can detect various facial expressions including:

- Raising eyebrows (Frontalis muscle)
- Frowning (Corrugator supercilii / Procerus muscles)
- Smiling (Zygomaticus major / minor muscles)
- Pouting (Orbicularis oris muscle)
- Squinting (Orbicularis oculi muscle)
- Showing disgust (Levator labii superioris muscle)
- Flaring nostrils (Nasalis muscle)
- Puckering chin (Mentalis muscle)
- Depressing mouth corners (Depressor anguli oris muscle)

## Output

The program generates two types of reports:

1. A JSON file containing detailed analysis data
2. A PDF report with visualizations and summaries

## Dependencies

Major dependencies include:

- OpenCV (cv2)
- MediaPipe
- NumPy
- ReportLab
- PyTTSx3
- SoundDevice

## Troubleshooting

If you encounter any issues:

1. Ensure your webcam is properly connected and accessible
2. Check that all dependencies are correctly installed
3. Verify that you have sufficient lighting for facial detection
4. Make sure you have appropriate permissions for camera access
