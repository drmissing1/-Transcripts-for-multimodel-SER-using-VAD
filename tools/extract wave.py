import os
import subprocess

root_dir = r""
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

count = 0

for subject in os.listdir(root_dir):
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for emotion in os.listdir(subject_path):
        emotion_path = os.path.join(subject_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for sentence in os.listdir(emotion_path):
            sentence_path = os.path.join(emotion_path, sentence)
            if not os.path.isdir(sentence_path):
                continue

            for file in os.listdir(sentence_path):
                if file.endswith(".avi"):
                    video_path = os.path.join(sentence_path, file)
                    output_name = f"{subject}_{emotion}_{sentence}.wav"
                    output_path = os.path.join(output_dir, output_name)

                    print(f" Found video: {video_path}")
                    print(f" Extracting to: {output_path}")

                    command = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        output_path
                    ]

                    result = subprocess.run(command)
                    if result.returncode != 0:
                        print(f" ffmpeg failed for: {video_path}")
                    else:
                        count += 1

print(f"\n processed {count} videos")