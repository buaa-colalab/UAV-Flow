import os
import argparse
import zipfile
import sys
import shutil
import unrealcv
modelscope = {
    'UE4': 'UnrealZoo/UnrealZoo-UE4',
    'UE5': 'UnrealZoo/UnrealZoo-UE5',
}
binary_linux = dict(
    UE4_ExampleScene='UE4_ExampleScene_Linux.zip',
    UE5_ExampleScene='UE5_ExampleScene_Linux.zip',
    UE4_Collection_Preview='Collection_v4_LinuxNoEditor.zip',
    Textures='Textures.zip'
)

binary_win = dict(
    UE4_ExampleScene='UE4_ExampleScene_Win.zip',
    UE5_ExampleScene='UE5_ExampleScene_Win.zip',
    Textures='Textures.zip'
)

binary_mac = dict(
    UE4_ExampleScene='UE4_ExampleScene_Mac.zip',
    Textures='Textures.zip'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='Textures',
                        help='Select the binary to download')
    parser.add_argument("-cloud", "--cloud", nargs='?', default='modelscope',
                        help='Select the cloud to download the binary, modelscope or aliyun')
    args = parser.parse_args()

    if 'linux' in sys.platform:
        binary_all = binary_linux
    elif 'darwin' in sys.platform:
        binary_all = binary_mac
    elif 'win' in sys.platform:
        binary_all = binary_win

    if args.env in binary_all:
        target_name = binary_all[args.env]
    else:
        print(f"{args.env} is not available to your platform")
        exit()

    if args.cloud == 'modelscope':
        if 'UE5' in target_name:
            remote_repo = modelscope['UE5']
        else:
            remote_repo = modelscope['UE4']
        cmd = f"modelscope download --dataset {remote_repo} --include {target_name} --local_dir ."
        try:
            os.system(cmd)
        except:
            print('Please install modelscope first: pip install modelscope')
            exit()
        filename = target_name

    with zipfile.ZipFile(filename, "r") as z:
        z.extractall()  # extract the zip file
    if 'Textures' in filename:
        folder ='textures'
    else:
        folder = filename[:-4]
    target = unrealcv.util.get_path2UnrealEnv()
    shutil.move(folder, target)
    os.remove(filename)

