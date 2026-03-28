import cv2
import os
from tqdm import tqdm

# //data
BASE_PATH = r"D:\projects__\hackstreet_26\datasets\final_dataset"

FIGHT_PATH = os.path.join(BASE_PATH, "fight")
NON_FIGHT_PATH = os.path.join(BASE_PATH, "non_fight")

OUTPUT_BASE = r"D:\projects__\hackstreet_26\datasets\processed"

FIGHT_OUT = os.path.join(OUTPUT_BASE, "fight")
NON_FIGHT_OUT = os.path.join(OUTPUT_BASE, "non_fight")


def extract_frames(video_path, output_folder, num_frames=16):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        return False

    step = total_frames // num_frames

    count = 0
    saved = 0

    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved}.jpg"), frame)
            saved += 1

        count += 1

    cap.release()
    return True


def process_folder(input_dir, output_dir, limit=None):
    os.makedirs(output_dir, exist_ok=True)

    videos = os.listdir(input_dir)

    if limit:
        videos = videos[:limit]

    success = 0
    failed = 0

    for video in tqdm(videos):
        video_path = os.path.join(input_dir, video)

        video_name = os.path.splitext(video)[0]
        save_path = os.path.join(output_dir, video_name)

        os.makedirs(save_path, exist_ok=True)

        ok = extract_frames(video_path, save_path)

        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} | Failed: {failed}")


# //test run
process_folder(FIGHT_PATH, FIGHT_OUT, limit=None)
process_folder(NON_FIGHT_PATH, NON_FIGHT_OUT, limit=None)