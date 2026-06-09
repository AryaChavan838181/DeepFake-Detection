import os
import json
import torch
from glob import glob
from tqdm import tqdm
from predict_pytorch import load_model, get_intel_backend, predict_video
import random

def evaluate_model(metadata_path, video_dir, ckpt_path="checkpoints/best_model.pth", max_videos=100, frames_per_video=10):
    print("\n" + "="*50)
    print("🚀 AUTOMATED DEEPFAKE MODEL EVALUATION")
    print("="*50)

    # 1. Load Metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 2. Setup Device & Model
    device, has_ipex = get_intel_backend()
    model = load_model(ckpt_path, device, has_ipex)

    # 3. Categorize Videos
    real_videos = []
    fake_videos = []

    for video_name, info in metadata.items():
        video_path = os.path.join(video_dir, video_name)
        if os.path.exists(video_path):
            if info['label'] == 'REAL':
                real_videos.append(video_path)
            else:
                fake_videos.append(video_path)

    # Balance the test set so accuracy isn't skewed (50% real / 50% fake)
    test_size = min(len(real_videos), len(fake_videos), max_videos // 2)
    
    if test_size == 0:
        print("❌ Not enough videos found in the specified directory!")
        return

    test_videos = []
    # Add tuples of (path, ground_truth_label, integer_label)
    # 0 = REAL, 1 = FAKE mapping used in PyTorch ImageFolder
    for v in random.sample(real_videos, test_size):
        test_videos.append((v, 'REAL', 0))
    for v in random.sample(fake_videos, test_size):
        test_videos.append((v, 'FAKE', 1))

    random.shuffle(test_videos)
    
    print(f"\n📊 Evaluating on {len(test_videos)} videos (Balanced: {test_size} REAL, {test_size} FAKE)...\n")

    # 4. Evaluation Loop
    correct_predictions = 0
    total_processed = 0
    false_positives = 0 # Predicted FAKE, was REAL
    false_negatives = 0 # Predicted REAL, was FAKE

    # Use tqdm progress bar
    pbar = tqdm(test_videos, desc="Testing Videos", unit="vid")

    # Suppress print statements from predict_video during the loop using contextlib if we wanted, 
    # but we'll let tqdm handle formatting
    import sys
    # 'os' is already imported at the top of the file

    for video_path, true_label_str, true_label_int in pbar:
        # We must allow emojis to be printed to os.devnull in Windows
        import io
        devnull = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            avg_prob = predict_video(video_path, model, device, num_frames=frames_per_video)
        except Exception as e:
            # Temporarily restore stdout just to print errors if they occur
            sys.stdout.close()
            sys.stdout = old_stdout
            print(f"\nError processing {video_path}: {e}")
            avg_prob = None
            sys.stdout = devnull
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        if avg_prob is None:
            continue # Face not found

        total_processed += 1
        predicted_fake = avg_prob < 0.5  # <--- CRITICAL FIX: ImageFolder gave FAKE Class 0 and REAL Class 1 during training!
        predicted_label_str = "FAKE" if predicted_fake else "REAL"

        if (predicted_fake and true_label_str == 'FAKE') or (not predicted_fake and true_label_str == 'REAL'):
            correct_predictions += 1
        else:
            if predicted_fake:
                false_positives += 1
            else:
                false_negatives += 1

        accuracy = (correct_predictions / total_processed) * 100
        pbar.set_postfix({"Acc": f"{accuracy:.1f}%"})

    # 5. Final Report
    if total_processed == 0:
        print("❌ No videos successfully processed.")
        return

    final_accuracy = (correct_predictions / total_processed) * 100

    print("\n\n" + "="*40)
    print("📈 EVALUATION RESULTS")
    print("="*40)
    print(f"Total Videos Tested:   {total_processed}")
    print(f"Overall Accuracy:      {final_accuracy:.2f}%\n")
    print(f"✅ Correct Guesses:    {correct_predictions}")
    print(f"❌ False Positives:    {false_positives} (Real videos called Fake)")
    print(f"❌ False Negatives:    {false_negatives} (Fake videos called Real)")
    print("="*40 + "\n")

    if final_accuracy < 60:
        print("💡 NOTE: An accuracy near 50-60% means the model is just guessing or biased.")
        print("   This is expected because you only trained on 5 frames per video.")
        print("   To fix this, re-run extraction with `--frames 15` and train longer.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="dfdc_train_part_00/dfdc_train_part_0/metadata.json")
    parser.add_argument("--video_dir", type=str, default="dfdc_train_part_00/dfdc_train_part_0/")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos to test (will be 50% real / 50% fake)")
    parser.add_argument("--frames", type=int, default=10, help="Frames to test per video")
    args = parser.parse_args()

    evaluate_model(args.metadata, args.video_dir, ckpt_path=args.ckpt, max_videos=args.max_videos, frames_per_video=args.frames)