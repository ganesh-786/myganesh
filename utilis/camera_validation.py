# import cv2
# import numpy as np
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import dlib
# import logging
# import base64
# from scipy.interpolate import interp1d
# import json  # Import the json module
# from django.conf import settings
# import os


# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Initialize Dlib for face and landmark detection
# detector = dlib.get_frontal_face_detector()
# predictor_path = "G:\Workventures\test - Copy\test\citizenship\static\models\shape_predictor_68_face_landmarks.dat"

# # Load the shape predictor within a try block
# try:
#     predictor = dlib.shape_predictor(predictor_path)
#     print("Dlib shape predictor loaded successfully.")
# except Exception as e:
#     print(f"Error loading predictor: {e}")
#     predictor = None  # Ensure predictor is None if loading fails


# def decode_image(base64_data):
#     """Decode base64 image data."""
#     try:
#         # Split the data URL at the comma and get the actual base64 data
#         base64_data = base64_data.split(",")[1]
#         img_data = base64.b64decode(base64_data)
#         img_array = np.frombuffer(img_data, np.uint8)
#         image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Corrected flag
#         if image is None:
#             raise ValueError("Failed to decode image.")
#         return image
#     except Exception as e:
#         logging.error(f"Image decoding failed: {str(e)}")
#         raise ValueError(f"Image decoding failed: {str(e)}")


# def check_blur(image):
#     """Detect if an image is blurry using Laplacian variance."""
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         is_blurry = laplacian_var < 150  # keep the same good value
#         return {
#             "status": not is_blurry,  # keep like this because I wanna test that it works
#             "message": "Image is clear." if not is_blurry else "Image is too blurry.",
#             "laplacian_variance": laplacian_var
#         }
#     except Exception as e:
#         logging.error(f"Blur check failed: {e}")
#         return {"status": False, "message": f"Blur check failed: {e}"}


# def check_brightness(image):
#     """Check if the image has adequate brightness."""
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         brightness = np.mean(gray)
#         is_bright =  40 <= brightness <= 230  # keep the same good value
#         return {
#             "status": is_bright,  # keep like this because I wanna test that it works
#             "message": "Brightness is adequate." if is_bright else "Image is too dark or too bright.",
#             "brightness": brightness
#         }
#     except Exception as e:
#         logging.error(f"Brightness check failed: {e}")
#         return {"status": False, "message": f"Brightness check failed: {e}"}


# def estimate_distance(image):
#     """Estimate the distance of the face from the camera using face width measurement."""
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)  # Detect faces

#         if len(faces) == 0:
#             return {"status": False, "message": "No face detected."}  # Now it correctly fails

#         if len(faces) > 1:  # Add this check - good practice
#             return {"status": False, "message": "Multiple faces detected."}

#         face = faces[0]  # Assume only one face for now
#         face_width = face.right() - face.left()  # Calculate the face width in pixels

#         # Pre-defined calibration data (face width in pixels vs. distance in cm)
#         calibration_data = {120: 25, 90: 35, 60: 50, 30: 80}

#         # Interpolate estimated distance
#         face_widths = np.array(list(calibration_data.keys()))
#         distances = np.array(list(calibration_data.values()))

#         # Interpolation function
#         f = interp1d(face_widths, distances, fill_value="extrapolate")
#         estimated_distance = float(f(face_width))  # Get the estimated distance and convert to float

#         # Ensure the distance is within a reasonable range
#         if 15 <= estimated_distance <= 130:  # Good thresholds
#             return {"status": True, "message": f"Face is at {estimated_distance:.2f} cm.", "distance": estimated_distance}
#         else:
#             return {"status": False, "message": "Face is too far or too close.", "distance": estimated_distance}
#     except Exception as e:
#         logging.error(f"Distance estimation failed: {e}")
#         return {"status": False, "message": f"Distance estimation failed: {e}"}


# def calculate_ear(eye):
#     """Calculate the Eye Aspect Ratio (EAR)."""
#     try:
#         A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
#         B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
#         C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
#         ear = (A + B) / (2.0 * C)
#         return ear
#     except Exception as e:
#         logging.error(f"EAR calculation failed: {e}")
#         return 0.0  # Return a default EAR value in case of an error


# def detect_blink(landmarks_list):
#     """Detect blinking by counting EAR drops below a threshold."""
#     try:
#         blink_count = 0
#         frame_counter = 0
#         EAR_THRESHOLD = 0.23  # Adjust this threshold as needed - less strict.  Good Value!
#         CONSECUTIVE_FRAMES = 2  # Adjust this value as needed - less strict. Good Value!

#         # Check if there are enough landmarks to process. If there are no landmarks at all, that means there's no face so, just fail for now
#         if not landmarks_list or len(landmarks_list) == 0:
#             return {"status": False, "message": "No face landmarks detected for blink detection."}


#         for landmarks in landmarks_list:
#             left_eye = landmarks[36:42]
#             right_eye = landmarks[42:48]
#             left_ear = calculate_ear(left_eye)
#             right_ear = calculate_ear(right_eye)
#             ear = (left_ear + right_ear) / 2.0
#             if ear < EAR_THRESHOLD:
#                 frame_counter += 1
#             else:
#                 if frame_counter >= CONSECUTIVE_FRAMES:
#                     blink_count += 1
#                 frame_counter = 0
#         return {
#             "status": blink_count >= 1,  # Only need one blink now  - makes sense
#             "message": f"{blink_count} blinks detected." if blink_count >= 1 else "Insufficient blinks detected.",
#             "blink_count": blink_count
#         }
#     except Exception as e:
#         logging.error(f"Blink detection failed: {e}")
#         return {"status": False, "message": f"Blink detection failed: {e}"}


# def extract_landmarks(frame):
#     """Extract facial landmarks from a single frame."""
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
#         if not faces:
#             return None
#         face = faces[0]
#         landmarks = predictor(gray, face)
#         return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
#     except Exception as e:
#         logging.error(f"Landmark extraction failed: {e}")
#         return None


# @csrf_exempt
# def validate_user(request):
#     """Handle photo capture, real-time checks, and liveness validation in one step."""
#     if request.method != "POST":
#         return JsonResponse({
#             "status": "error",
#             "message": "Invalid request method. Please use POST."
#         }, status=405)  # Method Not Allowed

#     try:
#         # Decode video frames
#         video_frames_data = request.FILES.getlist("video_frames[]")
#         if not video_frames_data:
#             return JsonResponse({
#                 "status": "error",
#                 "message": "No video frames provided."
#             }, status=400)  # Bad Request

#         # Decode frames from binary data
#         video_frames = []
#         for frame_file in video_frames_data:
#             nparr = np.frombuffer(frame_file.read(), np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             if frame is None:
#                 raise ValueError("Failed to decode one or more frames.")
#             video_frames.append(frame)

#         # Perform real-time checks on the first frame
#         first_frame = video_frames[0]
#         blur_check = check_blur(first_frame)
#         brightness_check = check_brightness(first_frame)
#         distance_check = estimate_distance(first_frame)

#         # Perform liveness check based on video frames
#         landmarks_list = []
#         for frame in video_frames:
#             landmarks = extract_landmarks(frame)
#             if landmarks:
#                 landmarks_list.append(landmarks)

#         blink_detected = detect_blink(landmarks_list) # No default message as this is dealt in the detect_blink function


#         # Combine all checks results
#         validation_results = {
#             "blur_check": blur_check,
#             "brightness_check": brightness_check,
#             "distance_check": distance_check,
#             "blink_detected": blink_detected,
#         }

#         # Determine overall status - MUST PASS ALL CHECKS.   If any fail, validation fails.
#         overall_status = all(result['status'] for result in validation_results.values())


#         # Construct response message.  Provide SPECIFIC failure reasons.
#         failure_reasons = []
#         if not blur_check['status']:
#             failure_reasons.append(blur_check['message'])
#         if not brightness_check['status']:
#             failure_reasons.append(brightness_check['message'])
#         if not distance_check['status']:
#             failure_reasons.append(distance_check['message'])
#         if not blink_detected['status']:
#             failure_reasons.append(blink_detected['message'])


#         if overall_status:
#             overall_message = "User validation passed."
#         else:
#             overall_message = "User validation failed: " + ", ".join(failure_reasons) # Specific reasons


#         # Create response data
#         response_data = {
#             "status": "success" if overall_status else "failure",
#             "message": overall_message,
#             "validation_results": validation_results,
#         }

#         # Convert numpy.bool_ to bool for JSON serialization
#         for key, value in response_data["validation_results"].items():
#             if isinstance(value, dict) and "status" in value:
#                 response_data["validation_results"][key]["status"] = bool(value["status"])


#         # Return JSON response
#         return JsonResponse(response_data)

#     except Exception as e:
#         logging.error(f"Error during validation: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": f"An error occurred: {str(e)}"
#         }, status=500)