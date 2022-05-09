import json
from tqdm import tqdm
import cv2

# data_types = ['ocr', 'htr']
data_types = ['wild']

obj_list = ['test', 'train', 'validation']
for data_type in data_types:
    data_root_path = f'./aihub_data/{data_type}/images/'
    save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'
    for obj in obj_list:
        # Load Save json file, Save gt file
        annotation_file = json.load(open(f'./{data_type}_{obj}_annotation.json'))
        gt_file = open(f'{save_root_path}gt_{obj}.txt', 'w')
        # 내부 객체 하나하나 읽음 (file 이름으로 읽음)
        for file_name in tqdm(annotation_file):
            # 객체의 파일 이름을 키로 값을 읽음 (annotation 객체)
            annotations = annotation_file[file_name]
            image = cv2.imread(f"{data_root_path}{file_name}")
            for idx, annotation in enumerate(annotations):
                # 어노테이션 객체의 텍스트 속성을 읽음
                text = annotation['text']                
                if "bbox" in annotation:
                    x, y, w, h = annotation['bbox']
                    if x <= 0 or y <= 0 or w <= 0 or h <= 0:
                        continue
                crop_img = image[y:y+h, x:x+w]
                crop_file_name = file_name[:-4] + '_{:03}.jpg'.format(idx + 1)
                cv2.imwrite(f"{save_root_path}{obj}/{crop_file_name}", crop_img)
                gt_file.write(f'{obj}/{file_name}\t{text}\n')