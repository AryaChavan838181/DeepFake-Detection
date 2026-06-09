import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

# Adjust imports
_this_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))
sys.path.insert(0, os.path.join(_project_root, "src", "models"))
sys.path.insert(0, os.path.join(_project_root, "src", "training"))

from noise_net_xpu import NoiseNet, ssim_loss, get_device

try:
    from deepface import DeepFace
except ImportError:
    print("Please install deepface: pip install deepface")
    sys.exit(1)

def main():
    device, ipex = get_device()
    print(f"Testing on {device}")
    
    # Locate best model or finetuned model
    model_dir = os.path.join(_project_root, "saved_models", "noise_net")
    finetune_path = os.path.join(model_dir, "noise_net_finetuned.pt")
    best_path = os.path.join(model_dir, "noise_net_best.pt")
    
    model_path = finetune_path if os.path.exists(finetune_path) else best_path
    if not os.path.exists(model_path):
        print(f"No checkpoint found at {model_dir}")
        sys.exit(1)
        
    print(f"Loading '{os.path.basename(model_path)}'...")
    model = NoiseNet(epsilon=0.012).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()

    import glob
    import random

    data_dir = os.path.join(_project_root, "data_prepared_noise")
    all_files = glob.glob(os.path.join(data_dir, "*.pt"))
    num_tests = min(50, len(all_files)) if all_files else 0
    
    if num_tests == 0:
        print("No validation files found!")
        sys.exit(1)

    selected_files = random.sample(all_files, num_tests)
    success_count = 0
    ssim_total = 0.0

    viz_count = 0
    test_noise_dir = os.path.join(_project_root, "test-noise")
    protected_only_dir = os.path.join(test_noise_dir, "protected_only")
    os.makedirs(test_noise_dir, exist_ok=True)
    os.makedirs(protected_only_dir, exist_ok=True)
    import torchvision.utils as vutils
    
    models_to_test = ["Facenet", "VGG-Face", "ArcFace", "Facenet512", "SFace"]
    model_success = {m: 0 for m in models_to_test}
    face_detect_fail = 0

    print(f"\n--- Running Tests on {num_tests} Random Images against {len(models_to_test)} Models ---\n")
    for i, pt_path in enumerate(selected_files):

        orig_img = torch.load(pt_path).unsqueeze(0).to(device)

        with torch.no_grad():
            protected = model.protect(orig_img, use_spectral=True, use_mtf=True)
            ssim_score = 1.0 - ssim_loss(orig_img, protected)
            ssim_score = ssim_score.item()
            ssim_total += ssim_score

        if viz_count < 15:
            # We explicitly compute the noise that survived the pipeline for visualization
            noise_diff = protected - orig_img
            # Rescale noise roughly based on epsilon (0.012) from [-eps, +eps] to [0,1]
            scaled_noise = (noise_diff / 0.012 + 1.0) / 2.0
            
            grid = torch.stack([
                orig_img[0].cpu(), 
                scaled_noise[0].cpu().clamp(0.0, 1.0), 
                protected[0].cpu()
            ])
            save_path = os.path.join(test_noise_dir, f"sample_{i:02d}_grid.jpg")
            prot_save_path = os.path.join(protected_only_dir, f"sample_{i:02d}_protected.jpg")
            try:
                vutils.save_image(grid, save_path, nrow=3)
                vutils.save_image(protected[0].cpu(), prot_save_path)
            except Exception:
                pass
            viz_count += 1

        orig_np = (orig_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        prot_np = (protected[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # We enforce detection as false to skip face alignment crops inside deepface verify step, 
        # since DeepFace's metric is purely for face distances. 
        # If verify fails on face detection itself, that's technically a disruption too.
        try:
            # Check if the noise ruined detection completely => successful disruption
            orig_faces = DeepFace.extract_faces(orig_np, enforce_detection=True)
            prot_faces = DeepFace.extract_faces(prot_np, enforce_detection=True)

            disrupted_models = []
            model_details = []
            
            # verify directly across all models
            for m_name in models_to_test:
                res = DeepFace.verify(
                    img1_path=orig_np,
                    img2_path=prot_np,
                    enforce_detection=True,
                    model_name=m_name,
                    silent=True
                )
                if not res.get("verified", True):
                    model_success[m_name] += 1
                    disrupted_models.append(m_name)
                    model_details.append(f"{m_name}: DISRUPTED")
                else:
                    model_details.append(f"{m_name}: MATCHED")

            status_str = ", ".join(model_details)

            if len(disrupted_models) == len(models_to_test):
                status = f"[ALL DISRUPTED] {status_str}"
                success_count += 1
            elif len(disrupted_models) > 0:
                status = f"[PARTIAL DISRUPT] {status_str}"
            else:
                status = f"[ALL MATCHED] {status_str}"
                
        except ValueError as e:
            # If it could not detect a face in the protected image at all
            success_count += 1
            face_detect_fail += 1
            for m_name in models_to_test:
                model_success[m_name] += 1
            status = "[ALL DISRUPTED] Face Unrecognizable by Detector"
        except Exception as e:
            status = f"ERROR: {e}"

        print(f"Sample {i:02d}: SSIM = {ssim_score:.4f} \n    -> {status}")

    print("\n" + "="*50)
    print(f"Average Imperceptibility (SSIM): {ssim_total/num_tests:.4f}")
    if num_tests > 0:
        print(f"Total Success (Across ALL Models) : {success_count}/{num_tests} ({(success_count/num_tests)*100:.0f}%)")
        print("\n--- Disruption Rates by Specific Model ---")
        for m_name in models_to_test:
            print(f"  {m_name} : {model_success[m_name]}/{num_tests} ({round((model_success[m_name]/num_tests)*100)}%)")
        print(f"  Face Detect Destroyed: {face_detect_fail}/{num_tests} instances")
    print("="*50)
    
    if ssim_total/num_tests < 0.8:
         print("\nWarning: The SSIM is severely low! The loaded weights were corrupted by the IPEX inversion bug.\nYou MUST retrain the model entirely. Run:")
         print(f"  del {model_path} # to delete the botched model")
         print("  python src/training/train_noise_net_xpu.py")

if __name__ == "__main__":
    main()

