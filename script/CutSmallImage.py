import os
import shutil
from PIL import Image

def copy_large_images(src_folder, dest_folder, min_width, min_height):
    """
    폴더 내에서 지정된 크기보다 큰 이미지를 다른 폴더로 복사하는 함수.
    
    :param src_folder: 이미지가 저장된 원본 폴더 경로
    :param dest_folder: 복사할 대상 폴더 경로
    :param min_width: 최소 너비
    :param min_height: 최소 높이
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # 대상 폴더가 없으면 생성
    
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 확장자 체크
            image_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width >= min_width and height >= min_height:
                        shutil.copy2(image_path, dest_path)  # 이미지 복사
                        print(f'Copied image: {filename}')
            except IOError as e:
                print(f"Error processing {filename}: {e}")

# 예시 사용법
copy_large_images('_CN/test_HR', '_CN/test_HR', 64, 64)  # 최소 크기 64x64 이상의 이미지를 복사