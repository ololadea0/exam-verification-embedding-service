# Facial Recognition System Main Entry Point
# This file serves as the main entry point for the facial recognition system

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function for the facial recognition system.
    This can be used for testing or running the system directly.
    """
    print("Facial Recognition System")
    print("Use embed.py for generating embeddings from images")
    print("Use recognition.py for face recognition tasks")
    print("Use capture.py for capturing training images")

if __name__ == "__main__":
    main()
