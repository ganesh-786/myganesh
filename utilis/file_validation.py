import os
from django.core.exceptions import ValidationError
import cv2
from PIL import Image
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def validate_file_size(file, max_size_kb=2048):
    """Validates the file size."""
    if file.size > max_size_kb * 1024:
        raise ValidationError(f"File size cannot exceed {max_size_kb} KB.")


def validate_image_dimensions(file, max_width=2000, max_height=2000):
    """Validates the image dimensions."""
    try:
        with Image.open(file) as img:
            width, height = img.size
            if width > max_width or height > max_height:
                raise ValidationError(
                    f"Image dimensions cannot exceed {max_width}x{max_height} pixels."
                )
    except Exception as e:
        raise ValidationError(f"Invalid image file: {str(e)}")


def check_blur(file, threshold=100):
    """Checks if the image is blurry using Laplacian variance."""
    try:
        # Read the file into a NumPy array
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValidationError("Failed to decode image.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < threshold:
            raise ValidationError(f"Image is too blurry. Laplacian variance: {laplacian_var:.2f} < {threshold}")
    except Exception as e:
        raise ValidationError(f"Blur check failed: {str(e)}")


def check_brightness(file, min_brightness=50, max_brightness=200):
    """Checks the brightness of the image."""
    try:
        # Read the file into a NumPy array
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValidationError("Failed to decode image.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if not (min_brightness < brightness < max_brightness):
            raise ValidationError(
                f"Image brightness is outside acceptable range: {min_brightness} < {brightness:.2f} < {max_brightness}"
            )
    except Exception as e:
        raise ValidationError(f"Brightness check failed: {str(e)}")


def validate_image(file, max_size_kb=2048, max_width=2000, max_height=2000, blur_threshold=100, min_brightness=50, max_brightness=200):
    """
    Performs comprehensive validation for image files.
    Validates file size, dimensions, blur, and brightness.
    """
    try:
        validate_file_size(file)
        file.seek(0)  # Reset file pointer after size validation
        validate_image_dimensions(file, max_width, max_height)
        file.seek(0)  # Reset file pointer after dimension validation
        check_blur(file, blur_threshold)
        file.seek(0)  # Reset file pointer after blur validation
        check_brightness(file, min_brightness, max_brightness)
    except ValidationError as e:
        raise e


@csrf_exempt
def citizenship_validate(request):
    """
    View to validate front and back images of citizenship documents.
    Expects two files: 'front' and 'back'.
    Returns JSON response indicating success or failure with detailed error messages.
    """
    if request.method == 'POST':
        # Extract uploaded files
        front_file = request.FILES.get('front')
        back_file = request.FILES.get('back')

        # Validate both files are present
        if not front_file or not back_file:
            return JsonResponse({'success': False, 'errors': ['Both front and back images are required.']})
            
        # Initialize a list to collect validation errors
        errors = []

        # Helper function to validate an image and collect errors
        def validate_and_collect_errors(file, side):
            try:
                validate_image(file)
            except ValidationError as e:
                errors.append(f"{side}: {str(e)}")
            except Exception as e:
                errors.append(f"{side}: {str(e)}")

        # Validate front image
        if front_file:
            validate_and_collect_errors(front_file, "Front Image")
        else:
            errors.append("Front image required")

        # Validate back image
        if back_file:
            validate_and_collect_errors(back_file, "Back Image")
        else:
            errors.append("Back image required")

        # If there are any errors, return them
        if errors:
            return JsonResponse({'success': False, 'errors': errors})

        # If both validations pass
        return JsonResponse({'success': True, 'message': 'Images passed validation!'})

    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'})