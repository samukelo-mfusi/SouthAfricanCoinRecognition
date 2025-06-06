import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
import pickle
import time
from tqdm import tqdm
import seaborn as sns


class SouthAfricanCoinRecognizer:

    def __init__(self, verbose=True):
        self.model = None
        self.scaler = StandardScaler()
        self.coin_classes = ['5c', '10c', '20c', '50c', 'R1', 'R2', 'R5']
        self.feature_vectors = []
        self.labels = []
        self.verbose = verbose
        self.trained = False

    def _log(self, message):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    #Bank Coin Image Preprocessing and Enhancement
    def preprocess_image(self, img_path):

        try:
            # Read image with error handling for corrupt files
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image at {img_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Dynamic noise reduction based on image characteristics
            if gray.std() > 30:  # Only blur if there's significant noise
                kernel_size = max(3, int(min(gray.shape) / 100))  # Dynamic kernel size
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            else:
                blurred = gray

            # Advanced contrast enhancement with automatic clip limit
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # Unsharp masking for edge enhancement
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
            sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

            return sharpened

        except Exception as e:
            self._log(f"Error preprocessing {img_path}: {str(e)}")
            return None

    #Bank Coin Segmentation
    def segment_coin(self, img):

        try:
            # Multi-level adaptive thresholding
            thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
            thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 3)
            combined_thresh = cv2.bitwise_and(thresh1, thresh2)

            # Morphological operations to clean up the thresholded image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find contours with hierarchy
            contours, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on multiple criteria
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)

                # Skip very small or very large areas
                if area < 500 or area > (img.shape[0] * img.shape[1] * 0.8):
                    continue

                # Calculate circularity and convexity
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 0

                # Combined shape score
                shape_score = 0.6 * circularity + 0.4 * convexity

                if shape_score > 0.7:  # Strong coin-like shape
                    valid_contours.append(cnt)

            # Create mask from the best contour
            mask = np.zeros_like(img)
            if valid_contours:
                # Select the largest valid contour
                main_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

                # Refine mask with edge-aware dilation
                refined_mask = cv2.dilate(mask, kernel, iterations=1)

                # Apply mask to get segmented coin
                segmented = cv2.bitwise_and(img, img, mask=refined_mask)

                return segmented, refined_mask, [main_contour]

            return img, mask, []  # Fallback if no good contours found

        except Exception as e:
            self._log(f"Error in segmentation: {str(e)}")
            return img, np.zeros_like(img), []

    #Bank Coin Features Extraction
    def extract_features(self, img, mask, contours):

        features = []

        try:
            # 1. Basic intensity statistics
            mean_val, std_val = cv2.meanStdDev(img, mask=mask)
            features.extend([mean_val[0][0], std_val[0][0]])

            # 2. Multi-scale LBP texture features
            for radius in [3, 5, 7]:  # Multiple scales capture different texture details
                n_points = 8 * radius
                lbp = local_binary_pattern(img, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                       range=(0, n_points + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)  # Normalize
                features.extend(hist)

            # 3. Advanced shape descriptors
            if contours:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                # Hu Moments (7 values)
                moments = cv2.moments(cnt)
                hu_moments = cv2.HuMoments(moments)
                hu_moments = np.log(np.abs(hu_moments) + 1e-9)  # Log scale

                # Additional shape features
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                aspect_ratio = ma / MA if MA > 0 else 0

                features.extend([
                    area, perimeter, circularity, aspect_ratio,
                    *hu_moments.flatten()
                ])
            else:
                features.extend([0] * 11)  # 3 basic + 7 Hu + 1 aspect ratio

            # 4. Edge-based features
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (mask.sum() + 1e-7)
            features.append(edge_density)

            return np.array(features)

        except Exception as e:
            self._log(f"Error in feature extraction: {str(e)}")
            return np.zeros(59)  # Fallback: zero vector matching expected feature size

    def load_dataset(self, data_dir):

        self._log(f"Loading dataset from {data_dir}")
        start_time = time.time()

        self.feature_vectors = []
        self.labels = []
        class_counts = {cls: 0 for cls in self.coin_classes}

        try:
            for class_idx, coin_class in enumerate(self.coin_classes):
                class_path = os.path.join(data_dir, coin_class)
                if not os.path.exists(class_path):
                    self._log(f"Warning: Missing directory for class {coin_class}")
                    continue

                image_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
                if not image_files:
                    self._log(f"Warning: No PNG images found for class {coin_class}")
                    continue

                self._log(f"Processing {len(image_files)} images for {coin_class}")

                for img_file in tqdm(image_files, desc=f"Processing {coin_class}", disable=not self.verbose):
                    img_path = os.path.join(class_path, img_file)

                    # Process image through pipeline
                    preprocessed = self.preprocess_image(img_path)
                    if preprocessed is None:
                        continue

                    segmented, mask, contours = self.segment_coin(preprocessed)
                    features = self.extract_features(segmented, mask, contours)

                    self.feature_vectors.append(features)
                    self.labels.append(class_idx)
                    class_counts[coin_class] += 1

            # Convert to numpy arrays
            X = np.array(self.feature_vectors)
            y = np.array(self.labels)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            self._log(f"Dataset loaded successfully in {time.time() - start_time:.2f}s")
            self._log(f"Class distribution: {class_counts}")

            return X_scaled, y

        except Exception as e:
            self._log(f"Error loading dataset: {str(e)}")
            return None, None


    #Bank Coin Classification
    def train_model(self, data_dir=None, X=None, y=None, test_size=0.2, cv_folds=5):

        try:
            # Load data if not provided
            if X is None or y is None:
                if data_dir is None:
                    raise ValueError("Either data_dir or X,y must be provided")
                X, y = self.load_dataset(data_dir)
                if X is None:
                    return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)

            self._log("\nTraining model with SVM (RBF kernel)...")

            # Train with automatically tuned parameters
            self.model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                             class_weight='balanced', random_state=42)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
            self._log(f"Cross-validation scores: {cv_scores}")
            self._log(f"Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

            # Final training
            self.model.fit(X_train, y_train)
            self.trained = True

            # Evaluation
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

            self._log("\nTraining Results:")
            self._log(f"Train accuracy: {accuracy_score(y_train, train_pred):.4f}")
            self._log(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")

            self._log("\nClassification Report:")
            self._log(classification_report(y_test, test_pred, target_names=self.coin_classes))

            # Confusion matrix visualization
            self.plot_confusion_matrix(y_test, test_pred)

            return True

        except Exception as e:
            self._log(f"Error in model training: {str(e)}")
            return False

    def plot_confusion_matrix(self, y_true, y_pred):
        #Detailed confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.coin_classes,
                    yticklabels=self.coin_classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        self._log("Confusion matrix saved to confusion_matrix.png")

    def save_model(self, model_path='coin_recognition_model.pkl'):
        #Save the trained model and scaler to disk
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'classes': self.coin_classes,
                    'trained': self.trained
                }, f)
            self._log(f"Model saved successfully to {model_path}")
            return True
        except Exception as e:
            self._log(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path='coin_recognition_model.pkl'):
        #Load a trained model from disk
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.coin_classes = data['classes']
                self.trained = data['trained']
            self._log(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            self._log(f"Error loading model: {str(e)}")
            return False

    def predict_coin(self, img_path, visualize=False):

        if not self.trained:
            raise ValueError("Model not trained or loaded")

        start_time = time.time()

        try:
            # Process image
            preprocessed = self.preprocess_image(img_path)
            if preprocessed is None:
                return None, None, 0

            segmented, mask, contours = self.segment_coin(preprocessed)
            features = self.extract_features(segmented, mask, contours)

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict
            class_idx = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            # Create confidence dictionary
            confidence = {cls: prob for cls, prob in zip(self.coin_classes, probabilities)}

            # Visualize if requested
            if visualize:
                self.visualize_processing(img_path, preprocessed, segmented, contours)

            processing_time = time.time() - start_time
            self._log(f"Prediction completed in {processing_time:.2f}s")

            return self.coin_classes[class_idx], confidence, processing_time

        except Exception as e:
            self._log(f"Error in prediction: {str(e)}")
            return None, None, 0

    def visualize_processing(self, img_path, preprocessed, segmented, contours):
        #Generate a visualization of the processing pipeline
        original = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Create mask visualization
        mask_viz = np.zeros_like(original)
        if contours:
            cv2.drawContours(mask_viz, contours, -1, (0, 255, 0), 3)
            cv2.drawContours(mask_viz, contours, -1, (0, 255, 0), -1)

        # Create plot
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(original_rgb)
        plt.title("1. Original Image")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(preprocessed, cmap='gray')
        plt.title("2. Preprocessed (Enhanced)")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(segmented, cmap='gray')
        plt.title("3. Segmented Coin")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.addWeighted(original_rgb, 0.7, mask_viz, 0.3, 0))
        plt.title("4. Detected Contour Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('processing_steps.png')
        plt.close()
        self._log("Processing visualization saved to processing_steps.png")


def main():
    # Initialize the recognizer with verbose output
    recognizer = SouthAfricanCoinRecognizer(verbose=True)

    # Path to my dataset, note:change path to your dataset
    data_dir = r"C:\Users\cc\Documents\South_African_Coin_Recog_System\data"

    # Train a new model
    if recognizer.train_model(data_dir):
        recognizer.save_model()

    # Or load a pre-trained model
    if recognizer.load_model():
        # Test prediction with visualization
        test_image = os.path.join(data_dir, "R5", "R5_1.png")
        if os.path.exists(test_image):
            class_name, confidence, _ = recognizer.predict_coin(test_image, visualize=True)
            print(f"\nPredicted coin: {class_name}")
            print("Confidence:")
            for coin, prob in confidence.items():
                print(f"  {coin}: {prob:.2%}")
        else:
            print(f"Test image not found at {test_image}")


if __name__ == "__main__":
    main()