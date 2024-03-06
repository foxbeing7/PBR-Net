import os

root_folder = './data/obj'


for subdir, dirs, files in os.walk(root_folder):
    # 如果用huuman数据集，加一个名字
    #obj_files = [f for f in files if f.endswith('00000.obj')]
    obj_files = [f for f in files if f.endswith('.obj')]
    if obj_files:

        obj_file = os.path.join(subdir, obj_files[0])


        mtl_file = obj_file.replace('.obj', '.mtl')


        if os.path.exists(mtl_file):
            # 构建参数字符串

            args_string = '--views 1 --obj "{}" --output_folder ./output/blender_data'.format(obj_file)

            # 构建命令行字符串
            command = 'blender --background --python render_blender.py -- {}'.format(args_string)

            # 执行命令行
            os.system(command)

            print(f'Rendering {obj_file} completed.')
        else:
            print(f'MTL file not found for {obj_file}. Skipping...')
