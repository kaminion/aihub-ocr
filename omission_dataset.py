import json
from tqdm import tqdm

data_type = 'htr'
save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'

obj_list = ['test', 'train', 'validation']
for obj in obj_list:
    # Load Save json file, Save gt file
    annotation_file = json.load(open(f'./{data_type}_{obj}_annotation.json'))
    gt_file = open(f'{save_root_path}gt_{obj}_annotation.json', 'w')
    # 내부 객체 하나하나 읽음 (file 이름으로 읽음)
    for file_name in tqdm(annotation_file):
        # 객체의 파일 이름을 키로 값을 읽음 (annotation 객체)
        annotations = annotation_file[file_name]
        for idx, annotation in enumerate(annotations):
            # 어노테이션 객체의 텍스트 속성을 읽음
            text = annotation['text']
            gt_file.write(f'{obj}/{file_name}\t{text}')