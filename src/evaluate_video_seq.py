import os
import json
import torch
import random
from glob import glob
from tqdm import tqdm
import argparse
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure custom predict scripts can be imported
from src.predict_video_seq import load_video_model, get_intel_backend, predict_video_sequence

def find_metadata_files(base_dirs):
    paths = []
    for d in base_dirs:
        for root, dirs, files in os.walk(d):
            if 'metadata.json' in files:
                paths.append(os.path.join(root, 'metadata.json'))
    return paths

def evaluate_video_model(metadata_dirs, ckpt_path="checkpoints_video/video_best_model.pth", max_videos=100, frames_per_video=8, use_openvino=False):
    print("\n" + "="*50)
    print("🚀 AUTOMATED 3D SEQUENCE MODEL EVALUATION")
    if use_openvino:
        print("⚡ ACCELERATED BY INTEL OPENVINO")
    print("="*50)

    # 1. Load Metadata Across Parts
    metadata_files = find_metadata_files(metadata_dirs)

    real_videos = []
    fake_videos = []

    for m in metadata_files:
        with open(m, 'r') as f:
            data = json.load(f)
            folder_path = os.path.dirname(m)
            for file_name, info in data.items():
                v_path = os.path.join(folder_path, file_name)
                if os.path.exists(v_path):
                    if info['label'] == 'REAL':
                        real_videos.append(v_path)
                    else:
                        fake_videos.append(v_path)

    # 2. Setup Device & Model
    device, has_ipex = get_intel_backend()
    model = load_video_model(ckpt_path, device, has_ipex, use_openvino=use_openvino, num_frames=frames_per_video)

    # 3. Balance the Test Set
    test_size = min(len(real_videos), len(fake_videos), max_videos // 2)
    
    if test_size == 0:
        print("❌ Not enough videos found in the specified directories!")
        return

    test_videos = []
    
    # In seq models: FAKE is 1, REAL is 0 explicitly
    for v in random.sample(real_videos, test_size):
        test_videos.append((v, 'REAL', 0.0))
    for v in random.sample(fake_videos, test_size):
        test_videos.append((v, 'FAKE', 1.0))

    random.shuffle(test_videos)
    
    print(f"\n📊 Evaluating LSTM Sequence Model on {len(test_videos)} videos (Balanced: {test_size} REAL, {test_size} FAKE)...\n")

    # 4. Evaluation Loop
    correct_predictions = 0
    total_processed = 0
    false_positives = 0 # Predicted FAKE, was REAL
    false_negatives = 0 # Predicted REAL, was FAKE

    pbar = tqdm(test_videos, desc="Testing Videos", unit="vid")

    import io

    for video_path, true_label_str, true_label_int in pbar:
        # Redirect stdout to silently analyze
        devnull = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
        old_stdout = sys.stdout
        sys.stdout = devnull
        
        try:
            prob = predict_video_sequence(video_path, model, device, num_frames=frames_per_video)
        except Exception as e:
            # Revert to print the error
            sys.stdout = old_stdout
            print(f"\nError processing {video_path}: {e}")
            prob = None
        finally:
            sys.stdout = old_stdout
            devnull.close()

        if prob is None:
            continue

        total_processed += 1
        predicted_fake = prob >= 0.5 

        if (predicted_fake and true_label_str == 'FAKE') or (not predicted_fake and true_label_str == 'REAL'):
            correct_predictions += 1
        else:
            if predicted_fake:
                false_positives += 1
            else:
                false_negatives += 1

        accuracy = (correct_predictions / total_processed) * 100
        pbar.set_postfix({"Acc": f"{accuracy:.1f}%"})

    if total_processed == 0:
        print("❌ No videos successfully processed.")
        return

    final_accuracy = (correct_predictions / total_processed) * 100

    print("\n\n" + "="*40)
    print("📈 LSTM TEMPORAL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Videos Tested:   {total_processed}")
    print(f"Overall Accuracy:      {final_accuracy:.2f}%\n")
    print(f"✅ Correct Guesses:    {correct_predictions}")
    print(f"❌ False Positives:    {false_positives} (Real videos called Fake)")
    print(f"❌ False Negatives:    {false_negatives} (Fake videos called Real)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs='+', default=["dfdc_train_part_00", "dfdc_train_part_01"], help="List of dataset folder paths")
    parser.add_argument("--ckpt", type=str, default="checkpoints_video/video_best_model.pth")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos to test (50% real / 50% fake)")
    parser.add_argument("--frames", type=int, default=8, help="Frames tracked per video")
    parser.add_argument("--openvino", action="store_true", help="Use OpenVINO for massive inference acceleration")
    args = parser.parse_args()

    # Automatically swap the default checkpoint to the OpenVINO XML if the flag is provided
    # and the user hasn't explicitly specified a different .xml file.
    ckpt = args.ckpt
    if args.openvino and ckpt == "checkpoints_video/video_best_model.pth":
        ckpt = "openvino_model/video_model.xml"

    evaluate_video_model(args.dirs, ckpt_path=ckpt, max_videos=args.max_videos, frames_per_video=args.frames, use_openvino=args.openvino)