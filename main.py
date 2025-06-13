# main.py
from setup import *
from database import *
from recognition import *

def main():
    # Bước 1: Tải thư viện và cài đặt model
    print("Step 1: Setting up environment and downloading model...")

    # Bước 2: Tạo database
    print("Step 2: Creating face database...")
    # Chạy logic tạo database (đã có trong database.py)

    # Bước 3: Nhận diện với người bất kỳ, vẽ khung và xác suất
    print("Step 3: Recognizing face with bounding box and confidence...")
    # Chạy logic nhận diện (đã có trong recognition.py)

if __name__ == "__main__":
    main()
