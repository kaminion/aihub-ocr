import json
import random
import os 
from tqdm import tqdm

data_types = ['ocr', 'htr']
labeling_file_names = ['printed_data_info.json', 'handwriting_data_info_clean.json']
json_files = []

for idx, [data_type, info_file] in enumerate(zip(data_types, labeling_file_names)):
    print(idx, data_type, info_file)
    ## Check Json File
    json_info = json.load(open(f'./aihub_data/{data_type}/{info_file}', encoding="UTF8"))

    ## Separate dataset - train / validation / test
    image_files = os.listdir(f'./aihub_data/{data_type}/images/')
    total = len(image_files)
    
    random.shuffle(image_files)

    n_train = int(len(image_files) * 0.6)
    n_validation = int(len(image_files) * 0.2)
    n_test = int(len(image_files) * 0.2)

    ## 위에서 load한 image file들 담음
    train_files = image_files[:n_train]
    validation_files = image_files[n_train:n_train + n_validation]
    test_files = image_files[-n_test:]

    ## Separate image id - train, validation, test
    train_img_ids = {}
    validation_img_ids = {}
    test_img_ids = {}

    ## JSON 파일에 있는 이미지 정보로 데이터셋의 id를 분류
    ## perform -> dict: {file_name: id}
    for image in json_info['images']:
        if image['file_name'] in train_files:
            train_img_ids[image['file_name']] = image['id']
        elif image['file_name'] in validation_files:
            validation_img_ids[image['file_name']] = image['id']
        elif image['file_name'] in test_files:
            test_img_ids[image['file_name']] = image['id']

    ## Annotations - train, validation, test 
    ## file_name: []
    train_annotations = {file_name: [] for file_name in train_img_ids}
    validation_annotations = {file_name: [] for file_name in validation_img_ids} 
    test_annotations = {file_name: [] for file_name in test_img_ids}

    ## dict는 for문 돌리면 key값 나옴
    ## id 찾기 위한 테이블 역할
    ## perform -> dict: {id: file_name}
    train_ids_img = {train_img_ids[file_name]: file_name for file_name in train_img_ids}
    validation_ids_img = {validation_img_ids[file_name]: file_name for file_name in validation_img_ids}
    test_ids_img = {test_img_ids[file_name]: file_name for file_name in test_img_ids}

    ## annotation id 값 검사 후 annotation 객체를 데이터셋에 따라 분류
    ## perform -> dict: {file_name: annotation}
    for idx, annotations in enumerate(json_info['annotations']):
        if idx % 5000 == 0:
            print(idx, '/', len(json_info['annotations']), 'processed')
        if annotations['image_id'] in train_ids_img:
            train_annotations[train_ids_img[annotations['image_id']]].append(annotations)
        elif annotations['image_id'] in validation_ids_img:
            validation_annotations[validation_ids_img[annotations['image_id']]].append(annotations)
        elif annotations['image_id'] in test_ids_img:
            test_annotations[test_ids_img[annotations['image_id']]].append(annotations)

    ## Write json files
    ## 각각 만든 annotation 정보를 다시 json으로 만듬
    with open(f'./{data_type}_train_annotation.json', 'w') as file:
        json.dump(train_annotations, file)

    with open(f'./{data_type}_validation_annotation.json', 'w') as file:
        json.dump(validation_annotations, file)
        
    with open(f'./{data_type}_test_annotation.json', 'w') as file:
        json.dump(test_annotations, file)

    ## Make gt_xxx.txt files 
    data_root_path = f'./aihub_data/{data_type}/images/'    
    save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'

    obj_list = ['test', 'train', 'validation']
    for obj in obj_list:
        # Load Save json file, Save gt file
        annotation_file = json.load(open(f'./{data_type}_{obj}_annotation.json'))
        # gt 파일은 txt
        gt_file = open(f'{save_root_path}gt_{obj}.txt', 'w')
        # 내부 객체 하나하나 읽음 (file 이름으로 읽음)
        for file_name in tqdm(annotation_file):
            # 객체의 파일 이름을 키로 값을 읽음 (annotation 객체)
            annotations = annotation_file[file_name]
            for idx, annotation in enumerate(annotations):
                # 어노테이션 객체의 텍스트 속성을 읽음
                text = annotation['text']
                gt_file.write(f'{obj}/{file_name}\t{text}\n')