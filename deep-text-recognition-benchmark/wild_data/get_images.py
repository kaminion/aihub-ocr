import shutil
import os

data_root_path = '../../aihub_data/wild/images/'
save_root_path = 'images/'

# copy images from dataset directory to current directory
shutil.copytree(data_root_path, save_root_path)

# 

# separate dataset : train, validation, test
obj_list = ['train', 'test', 'validation']
for obj in obj_list:
  if(os.path.exists(os.path.join('./', obj)) == False):
    os.mkdir(os.path.join('./', obj))

  with open(f'gt_{obj}.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      file_path = line.split('.jpg')[0]
      file_name = file_path.split('/')[1] + '.jpg'
      
      # 파일 없을 때만 이동, 덮어쓰기 가능하게 변경
      if(os.path.exists(os.path.join(save_root_path, file_name))):
        res = shutil.move(os.path.join(save_root_path, file_name), 
          os.path.join('./', f'./{obj}/{file_name}'))
        