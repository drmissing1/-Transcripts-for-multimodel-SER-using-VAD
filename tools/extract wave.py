# import os
# import subprocess

# root_dir = r"D:\Multimodel SER system\project2_database\enterface database"
# output_dir = r"D:\Multimodel SER system\project2_database\enterface wave"
# os.makedirs(output_dir, exist_ok=True)

# count = 0

# for subject in os.listdir(root_dir):
#     subject_path = os.path.join(root_dir, subject)
#     if not os.path.isdir(subject_path):
#         continue

#     for emotion in os.listdir(subject_path):
#         emotion_path = os.path.join(subject_path, emotion)
#         if not os.path.isdir(emotion_path):
#             continue

#         for sentence in os.listdir(emotion_path):
#             sentence_path = os.path.join(emotion_path, sentence)
#             if not os.path.isdir(sentence_path):
#                 continue

#             for file in os.listdir(sentence_path):
#                 if file.endswith(".avi"):
#                     video_path = os.path.join(sentence_path, file)
#                     output_name = f"{subject}_{emotion}_{sentence}.wav"
#                     output_path = os.path.join(output_dir, output_name)

#                     print(f"🎥 Found video: {video_path}")
#                     print(f"🎧 Extracting to: {output_path}")

#                     command = [
#                         "ffmpeg",
#                         "-i", video_path,
#                         "-vn",
#                         "-acodec", "pcm_s16le",
#                         "-ar", "16000",
#                         "-ac", "1",
#                         output_path
#                     ]

#                     result = subprocess.run(command)
#                     if result.returncode != 0:
#                         print(f"❌ ffmpeg failed for: {video_path}")
#                     else:
#                         count += 1

# print(f"\n✅ 提取完成，共处理了 {count} 个视频")

# mp4_to_mp3_one.py
# 将单个 MP4 提取为同名 MP3（保存在同一目录）
# 依赖：ffmpeg 已安装且在 PATH 中

import subprocess
import shutil
from pathlib import Path
import sys

# 在这里填你的文件完整路径
INPUT_MP4 = Path(r"D:\桌面\大学资料\大四上\时事论坛.mp4")

def check_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        print("未找到 ffmpeg，请先安装并加入 PATH。")
        sys.exit(1)
    return exe

def main():
    if not INPUT_MP4.exists():
        print(f"找不到文件：{INPUT_MP4}")
        sys.exit(1)

    # 若你给的是无扩展名文件，这里补 .mp4；若已有扩展名会保持不变
    in_path = INPUT_MP4 if INPUT_MP4.suffix else INPUT_MP4.with_suffix(".mp4")
    if not in_path.exists():
        print(f"找不到文件（已尝试加 .mp4）：{in_path}")
        sys.exit(1)

    if in_path.suffix.lower() != ".mp4":
        print(f"输入文件不是 .mp4：{in_path}")
        sys.exit(1)

    ffmpeg = check_ffmpeg()
    out_path = in_path.with_suffix(".mp3")
    tmp_out = out_path.with_suffix(".tmp.mp3")

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(in_path),
        "-vn",
        "-map", "a?",
        "-acodec", "libmp3lame",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-map_metadata", "0",
        str(tmp_out),
    ]

    print(f"开始转换：{in_path.name} -> {out_path.name}")
    result = subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",   # 强制用 UTF-8
    errors="ignore"     # 有坏字节就忽略，避免 UnicodeDecodeError
)


    if result.returncode != 0:
        if "matches no streams" in result.stderr or "Stream specifier 'a?'" in result.stderr:
            print("该视频没有音频流，已跳过。")
        else:
            print("转换失败：")
            print(result.stderr)
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        sys.exit(1)

    if out_path.exists():
        out_path.unlink()
    tmp_out.rename(out_path)
    print(f"完成：{out_path}")

if __name__ == "__main__":
    main()
