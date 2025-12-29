import os
import subprocess


files=os.listdir("videos")
print(files)


for file in files:
    clean_name =os.path.splitext(file.split("  ")[0])[0]
    print(clean_name)
    subprocess.run(['ffmpeg', '-i',f'videos/{file}', '-vn', f'audio/{clean_name}.mp3'])