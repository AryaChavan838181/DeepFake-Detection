from mtcnn import MTCNN
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, target_size=(226,226)):
        self.target_size = target_size
        self.detector = MTCNN()

    def detect_face(self, image):
        # detecting faces and returning a bounding box
        result = self.detector.detect_faces(image)
        if not result:
            return None
        return result[0] #first face
    
    def preprocess_single(self, image):
        #will detect face using the above function, crop to bounding box, resize to 226x226 and normalize
        face_data = self.detect_face(image)

        if face_data is None:
            return None #no face was found
        
        x, y, w, h = face_data['box'] #coordinates, width, height
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, self.target_size)

        #normalizing using imagenet
        #imagenet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        face_normalized = face_resized.astype(np.float32)/255.0
        face_normalized = (face_normalized - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])

        return face_normalized
        
    def preprocess_batch(self, images):
        batch = []
        for img in images:
            processed = self.preprocess_single(img)
            if processed is not None:
                batch.append(processed)
        return np.array(batch)
    
    def _get_imagenet_stats(self):
        #just storing mean and std just in case we need them later
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return mean, std
    

# if __name__ == "__main__":
#     processor = ImageProcessor()
    
#     # Load a test image (replace with your actual image path)
#     test_image = cv2.imread("test_image.jpeg")
    
#     if test_image is None:
#         print("Error: Could not load image. Check the file path.")
#     else:
#         print(f"Image loaded. Shape: {test_image.shape}")
        
#         # Preprocess
#         result = processor.preprocess_single(test_image)
        
#         if result is not None:
#             print(f"✅ Success! Preprocessed shape: {result.shape}")
#             print(f"   Value range: [{result.min():.2f}, {result.max():.2f}]")
#         else:
#             print("❌ No face detected in image")
