"""
Solar Panel Bounding Box to Binary Mask Converter
==================================================
Extracts red bounding boxes (including rotated/non-uniform rectangles) from annotated 
satellite images and converts them to binary masks. Handles annotation text labels.

Requirements:
    pip install opencv-python numpy pillow matplotlib

Usage:
    python convert_bbox_to_mask.py

Directory Structure Expected:
    project/
    ├── original/          # Original satellite images
    ├── annotations/       # Annotated images with red bounding boxes
    └── masks/            # Output binary masks (created automatically)
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt


class BBoxToMaskConverter:
    """Converts red bounding boxes (including rotated rectangles) in annotated images to binary masks."""
    
    def __init__(self, original_dir, annotations_dir, output_dir, log_file="processing_log.json"):
        """
        Initialize the converter.
        
        Args:
            original_dir: Path to folder with original satellite images
            annotations_dir: Path to folder with annotated images (red bounding boxes)
            output_dir: Path to save binary masks
            log_file: Name of the log file to save processing results
        """
        self.original_dir = Path(original_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.log_file = log_file
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✓ Converter initialized")
        print(f"  Original images: {self.original_dir}")
        print(f"  Annotated images: {self.annotations_dir}")
        print(f"  Output masks: {self.output_dir}")
    
    def extract_red_mask(self, image):
        """
        Extract red bounding boxes using HSV color segmentation.
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Binary mask of red pixels
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color in HSV
        # Red wraps around in HSV (0-10 and 170-180 degrees)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return red_mask
    
    def remove_text_regions(self, red_mask, min_noise_area=20):
        """
        Remove text annotations from the red mask.
        Text/labels are NOT rectangular, while bounding boxes ARE rectangular.
        We keep only rectangular contours and remove everything else.
        
        Args:
            red_mask: Binary mask of red pixels
            min_noise_area: Minimum area to consider (removes tiny noise)
        Returns:
            Mask with only rectangular regions (boxes) kept
        """
        # Find all contours
        contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a clean mask (start with black)
        clean_mask = np.zeros_like(red_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small (noise)
            if area < min_noise_area:
                continue
            
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            is_rectangle = len(approx) == 4
            
            # Alternative check: use minimum area rectangle and compare areas
            # If the contour closely matches a rectangle, keep it
            if not is_rectangle:
                rect = cv2.minAreaRect(contour)
                rect_area = rect[1][0] * rect[1][1]  # width * height
                contour_area = cv2.contourArea(contour)
                
                # If contour area is very close to rectangle area, it's rectangular
                area_ratio = contour_area / rect_area if rect_area > 0 else 0
                is_rectangle = area_ratio > 0.85  # 85% match means it's rectangular
            
            # Keep only rectangular contours (these are bounding boxes)
            # Remove everything else (text, irregular shapes)
            if is_rectangle:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        return clean_mask
    
    def extract_polygon_boxes(self, red_mask, min_area=100):
        """
        Detect bounding boxes including rotated/non-uniform rectangles.
        Uses polygon approximation to handle any quadrilateral shape.
        
        Args:
            red_mask: Binary mask of red pixels (with text removed)
            min_area: Minimum area threshold to filter noise
            
        Returns:
            List of polygons as numpy arrays of corner points
        """
        # Morphological operations to connect box edges
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        processed = cv2.morphologyEx(processed, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            # Lower epsilon for more precise fitting
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we get a quadrilateral (4 corners), use it directly
            if len(approx) == 4:
                polygons.append(approx.reshape(-1, 2))
            # Otherwise, use the minimum area rectangle (handles rotated boxes)
            else:
                # Get rotated rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int8(box)
                polygons.append(box)
        
        return polygons
    
    def create_binary_mask_from_polygons(self, image_shape, polygons):
        """
        Create binary mask from polygons (including rotated rectangles).
        
        Args:
            image_shape: Shape of the original image (h, w, c)
            polygons: List of polygon corner points
            
        Returns:
            Binary mask (0 for background, 255 for solar panels)
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill each polygon region with white
        for polygon in polygons:
            polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)
        
        return mask
    
    def process_with_edge_detection(self, red_mask):
        """
        Alternative method: Use edge detection to find box boundaries.
        Useful for boxes that are drawn as outlines.
        
        Args:
            red_mask: Binary mask of red pixels
            
        Returns:
            Enhanced mask with connected edges
        """
        # Detect edges
        edges = cv2.Canny(red_mask, 50, 150)
        
        # Dilate edges to connect them
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        return closed
    
    def process_single_image(self, image_name, debug=False):
        """
        Process a single image pair to generate binary mask.
        Handles rotated/non-uniform rectangles and removes text annotations.
        
        Args:
            image_name: Name of the image file
            debug: If True, save intermediate processing steps
            
        Returns:
            Tuple of (success: bool, num_boxes: int, message: str)
        """
        original_path = self.original_dir / image_name
        annotated_path = self.annotations_dir / image_name
        output_path = self.output_dir / image_name
        
        # Check if files exist
        if not annotated_path.exists():
            return False, 0, f"Annotated image not found: {image_name}"
        
        try:
            # Load annotated image
            annotated_img = cv2.imread(str(annotated_path))
            
            if annotated_img is None:
                return False, 0, f"Failed to load image: {image_name}"
            
            # Step 1: Extract all red pixels
            red_mask = self.extract_red_mask(annotated_img)
            
            if debug:
                cv2.imwrite(str(self.output_dir / f"debug_1_red_mask_{image_name}"), red_mask)
            
            # Step 2: Remove text annotations
            clean_mask = self.remove_text_regions(red_mask)
            
            if debug:
                cv2.imwrite(str(self.output_dir / f"debug_2_clean_mask_{image_name}"), clean_mask)
            
            # Step 3: Try edge detection enhancement
            edge_enhanced = self.process_with_edge_detection(clean_mask)
            combined_mask = cv2.bitwise_or(clean_mask, edge_enhanced)
            
            if debug:
                cv2.imwrite(str(self.output_dir / f"debug_3_combined_{image_name}"), combined_mask)
            
            # Step 4: Extract polygon boxes (including rotated rectangles)
            polygons = self.extract_polygon_boxes(combined_mask)
            
            # Step 5: Create binary mask from polygons
            binary_mask = self.create_binary_mask_from_polygons(annotated_img.shape, polygons)
            
            # Save mask
            cv2.imwrite(str(output_path), binary_mask)
            
            num_boxes = len(polygons)
            
            if num_boxes == 0:
                return True, 0, f"Warning: No bounding boxes detected in {image_name}"
            
            return True, num_boxes, f"Successfully processed {image_name} - {num_boxes} box(es) found"
            
        except Exception as e:
            return False, 0, f"Error processing {image_name}: {str(e)}"
    
    def visualize_result(self, image_name, save_visualization=True):
        """
        Create visualization comparing original, annotated, and generated mask.
        Shows polygon overlays on the mask.
        
        Args:
            image_name: Name of the image to visualize
            save_visualization: Whether to save the visualization
        """
        original_path = self.original_dir / image_name
        annotated_path = self.annotations_dir / image_name
        mask_path = self.output_dir / image_name
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image (if exists)
        if original_path.exists():
            original = cv2.imread(str(original_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[0].imshow(original)
            axes[0].set_title("Original Image")
        else:
            axes[0].text(0.5, 0.5, "Original not available", ha='center', va='center')
            axes[0].set_title("Original Image")
        
        # Annotated image
        if annotated_path.exists():
            annotated = cv2.imread(str(annotated_path))
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            axes[1].imshow(annotated)
            axes[1].set_title("Annotated (Red Boxes + Text)")
        
        # Red mask extraction
        if annotated_path.exists():
            red_mask = self.extract_red_mask(cv2.imread(str(annotated_path)))
            axes[2].imshow(red_mask, cmap='gray')
            axes[2].set_title("Red Pixel Detection")
        
        # Generated mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            axes[3].imshow(mask, cmap='gray')
            axes[3].set_title("Generated Binary Mask")
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_visualization:
            viz_path = self.output_dir / f"viz_{image_name}"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {viz_path}")
        
        plt.show()
    
    def process_all_images(self, visualize_samples=3, debug_mode=False):
        """
        Process all images in the annotations directory.
        
        Args:
            visualize_samples: Number of sample results to visualize
            debug_mode: If True, save intermediate processing steps for debugging
        """
        # Get list of annotation images
        annotation_files = list(self.annotations_dir.glob("*.jpg")) + \
                          list(self.annotations_dir.glob("*.png")) + \
                          list(self.annotations_dir.glob("*.jpeg"))
        
        total_images = len(annotation_files)
        
        if total_images == 0:
            print(f"❌ No images found in {self.annotations_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing {total_images} images...")
        if debug_mode:
            print("DEBUG MODE: Saving intermediate processing steps")
        print(f"{'='*60}\n")
        
        for idx, img_path in enumerate(annotation_files, 1):
            image_name = img_path.name
            success, num_boxes, message = self.process_single_image(image_name, debug=debug_mode)
            
            # Update statistics
            self.stats["total_processed"] += 1
            if success:
                self.stats["successful"] += 1
                if num_boxes == 0:
                    self.stats["warnings"].append(message)
            else:
                self.stats["failed"] += 1
                self.stats["warnings"].append(message)
            
            # Print progress
            status = "✓" if success else "✗"
            print(f"[{idx}/{total_images}] {status} {message}")
        
        # Save log
        self.save_log()
        
        # Print summary
        self.print_summary()
        
        # Visualize samples
        if visualize_samples > 0 and self.stats["successful"] > 0:
            print(f"\nGenerating {min(visualize_samples, len(annotation_files))} sample visualizations...")
            for img_path in annotation_files[:visualize_samples]:
                self.visualize_result(img_path.name, save_visualization=True)
    
    def save_log(self):
        """Save processing log to JSON file."""
        log_path = self.output_dir / self.log_file
        with open(log_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n✓ Processing log saved: {log_path}")
    
    def print_summary(self):
        """Print processing summary statistics."""
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['warnings']:
            print(f"\nWarnings/Errors ({len(self.stats['warnings'])}):")
            for warning in self.stats['warnings'][:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.stats['warnings']) > 10:
                print(f"  ... and {len(self.stats['warnings']) - 10} more")
        
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print("Solar Panel Bounding Box to Binary Mask Converter")
    print("Handles: Rotated rectangles + Text annotations")
    print("="*60 + "\n")
    
    # Configure paths - MODIFY THESE TO MATCH YOUR DIRECTORY STRUCTURE
    original_dir = "data/processed/ORIGINAL"           # Folder with original satellite images
    annotations_dir = "data/processed/annotated_images"     # Folder with annotated images (red boxes)
    output_dir = "data/processed/masks_2"                # Output folder for binary masks
    
    # Initialize converter
    converter = BBoxToMaskConverter(
        original_dir=original_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir
    )
    
    # Process all images
    # Set debug_mode=True to save intermediate steps for troubleshooting
    converter.process_all_images(visualize_samples=3, debug_mode=False)
    
    print("\n✓ Processing complete!")
    print(f"✓ Binary masks saved to: {output_dir}/")
    print(f"✓ Visualizations saved to: {output_dir}/viz_*.jpg")
    print(f"✓ Processing log: {output_dir}/processing_log.json")
    print(f"\nTip: Set debug_mode=True in the script to see intermediate processing steps")


if __name__ == "__main__":
    main()