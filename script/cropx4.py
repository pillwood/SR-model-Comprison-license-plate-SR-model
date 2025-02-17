import os
from PIL import Image

def crop_to_nearest_multiple_of_4(image):
    # 이미지 크기 얻기
    width, height = image.size
    
    # 4의 배수로 맞추기 위해 크기 조정
    new_width = width - (width % 4)
    new_height = height - (height % 4)
    
    # 자르기
    cropped_image = image.crop((0, 0, new_width, new_height))
    
    return cropped_image

def process_images_in_folder(input_folder_path, output_folder_path):
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 폴더 내 모든 이미지 파일 처리
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 확장자 체크
            image_path = os.path.join(input_folder_path, filename)
            with Image.open(image_path) as img:
                # 4의 배수가 아니면 크기 조정
                cropped_img = crop_to_nearest_multiple_of_4(img)
                
                # 수정된 이미지를 새로운 파일로 저장 (파일명에 '_x4' 추가)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder_path, f'{name}_x4{ext}')
                cropped_img.save(output_path)
                print(f'Processed and saved: {output_path}')

# 폴더 경로를 입력하세요
process_images_in_folder('_CN/test_HR','_CN/test_HR')