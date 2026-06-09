import os
from PIL import Image, ImageDraw

def create_icons(source_image_path, res_dir):
    try:
        img = Image.open(source_image_path).convert("RGBA")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    sizes = {
        "mipmap-mdpi": 48,
        "mipmap-hdpi": 72,
        "mipmap-xhdpi": 96,
        "mipmap-xxhdpi": 144,
        "mipmap-xxxhdpi": 192,
    }

    for dir_name, size in sizes.items():
        target_dir = os.path.join(res_dir, dir_name)
        os.makedirs(target_dir, exist_ok=True)

        # Create standard square/rounded-square icon
        resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
        resized_img.save(os.path.join(target_dir, "ic_launcher.png"))

        # Create circular round icon
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        
        round_img = resized_img.copy()
        round_img.putalpha(mask)
        round_img.save(os.path.join(target_dir, "ic_launcher_round.png"))

    print("Successfully generated mipmap icons.")

if __name__ == "__main__":
    src_path = r"C:\Users\ritik\.gemini\antigravity\brain\df393be0-8ee3-402e-bb89-ce5b5dc54117\dfdetective_app_icon_1779481699734.png"
    res_path = r"c:\Users\ritik\Desktop\Projects\sem4_edi\android_app\app\src\main\res"
    create_icons(src_path, res_path)
