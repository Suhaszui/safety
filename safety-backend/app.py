import time
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from geopy.geocoders import Nominatim
import firebase_admin
from firebase_admin import credentials, messaging

# ==============================================================================
# --- 1. INITIALIZATION AND SETUP ---
# ==============================================================================

app = Flask(__name__)
CORS(app)

# --- Initialize Firebase Admin SDK ---
try:
    # IMPORTANT: Make sure 'firebase-adminsdk.json' is in the same folder.
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"🔴 ERROR: Could not initialize Firebase. Check your key file. Error: {e}")

# --- Initialize Geocoder for Reverse Geocoding ---
try:
    geolocator = Nominatim(user_agent="visakhapatnam_safety_app")
    print("✅ Geopy geocoder initialized successfully.")
except Exception as e:
    print(f"🔴 ERROR: Could not initialize geocoder: {e}")
    geolocator = None

# --- Load AI Model and District Data ---
try:
    model_pipeline = joblib.load("safety_model_pipeline.joblib")
    print("✅ AI model loaded successfully.")
except FileNotFoundError:
    print("🔴 ERROR: 'safety_model_pipeline.joblib' not found. Please run the model training script.")
    model_pipeline = None

try:
    # Use the district name as the index for faster lookups.
    district_data = pd.read_csv("cleaned_combined_data.csv").set_index("district_name")
    print("✅ District dataset loaded successfully.")
except FileNotFoundError:
    print("🔴 ERROR: 'crime_dataset.csv' not found in the backend folder.")
    district_data = None

# --- In-Memory State Management ---
# This dictionary acts as the server's short-term memory for active user trips.
# Format: {'user_id': 'last_known_district_name'}
USER_LAST_DISTRICT = {}


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def get_district_from_coords(lat, lon):
    """
    Performs real reverse geocoding using geopy and the Nominatim (OpenStreetMap) service.
    """
    if not geolocator:
        return None
    try:
        # Respect Nominatim's usage policy (max 1 request/sec) by sleeping for a second.
        time.sleep(1) 
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        
        if location:
            address = location.raw.get('address', {})
            # We check a list of common keys for the district name. 'county' is often used for districts in India.
            district = address.get('county', address.get('city_district', address.get('suburb', address.get('city', None))))
            return district
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

def interpret_score(score):
    """
    Takes a numerical AI score and translates it into the 5-Level Safety Alert System for the user.
    """
    display_score = round(score, 1)
    if score >= 95:
        return {"level": 5, "level_name": "All Clear", "tagline": f"Excellent Safety ({display_score}/100)", "color_code": "GREEN", "emoji": "🟢", "advice": "Extremely low risk. Feel free to explore and enjoy all activities with confidence."}
    elif 80 <= score < 95:
        return {"level": 4, "level_name": "Low Risk", "tagline": f"Good Safety ({display_score}/100)", "color_code": "BLUE", "emoji": "🔵", "advice": "Low risk profile. Practice standard awareness, like being mindful of belongings in crowded places."}
    elif 60 <= score < 80:
        return {"level": 3, "level_name": "Elevated Awareness", "tagline": f"Moderate Safety ({display_score}/100)", "color_code": "YELLOW", "emoji": "🟡", "advice": "Be observant. After 9:00 PM, stick to main roads and consider using a trusted taxi service for late travel."}
    elif 40 <= score < 60:
        return {"level": 2, "level_name": "High Caution", "tagline": f"High Risk ({display_score}/100)", "color_code": "ORANGE", "emoji": "🟠", "advice": "Limit non-essential travel, especially walking alone after sunset. Share your live location with a contact."}
    else:  # score < 40
        return {"level": 1, "level_name": "Critical Alert", "tagline": f"Critical Risk ({display_score}/100)", "color_code": "RED", "emoji": "🔴", "advice": "Very high-risk zone. Travel for essential purposes only, using secure transport. Do not travel alone."}

def send_push_notification(fcm_token, message_details):
    """
    Sends a real push notification to a specific device using Firebase Cloud Messaging (FCM).
    """
    try:
        # Construct the message payload for FCM.
        message = messaging.Message(
            # The 'notification' part is what the user sees when the app is in the background.
            notification=messaging.Notification(
                title=f"{message_details['emoji']} Safety Alert: {message_details['level_name']}",
                body=message_details['tagline'],
            ),
            # The 'data' payload is sent to your app to handle programmatically.
            data={
                'level': str(message_details['level']),
                'level_name': message_details['level_name'],
                'tagline': message_details['tagline'],
                'color_code': message_details['color_code'],
                'advice': message_details['advice'],
            },
            token=fcm_token, # The unique registration token of the target device.
        )
        
        # Send the message via the Firebase Admin SDK.
        response = messaging.send(message)
        print("🚀 Successfully sent push notification:", response)
    except Exception as e:
        print(f"🔴 ERROR sending push notification: {e}")


# ==============================================================================
# --- 3. API ENDPOINTS ---
# ==============================================================================

@app.route('/location_ping', methods=['POST'])
def location_ping():
    """
    Handles real-time location pings from the Flutter app, detects district changes,
    and triggers safety alerts.
    """
    # Check if all server components are loaded and ready.
    if not all([model_pipeline, district_data, geolocator]):
        return jsonify({'status': 'error', 'message': 'Server is not properly configured'}), 503

    data = request.get_json()
    # The Flutter app must send its FCM token with each ping.
    if not data or 'user_id' not in data or 'lat' not in data or 'lon' not in data or 'fcm_token' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data provided'}), 400

    user_id, lat, lon, fcm_token = data['user_id'], data['lat'], data['lon'], data['fcm_token']
    
    # 1. Determine the user's current district from their coordinates.
    current_district = get_district_from_coords(lat, lon)
    if not current_district:
        return jsonify({'status': 'ok', 'message': 'User is outside a recognizable district'}), 200

    # 2. Get the user's last known district from memory.
    last_known_district = USER_LAST_DISTRICT.get(user_id)

    # 3. Detect a "state change": if the user has entered a new district.
    if current_district != last_known_district:
        print(f"STATE CHANGE: User '{user_id}' entered '{current_district}' from '{last_known_district}'. Triggering alert.")
        
        try:
            # 4. Look up data for the new district.
            district_info = district_data.loc[current_district]
            
            # 5. Run the AI model to get a safety score.
            model_input = pd.DataFrame([{'crime_rate_per_capita': district_info['crime_rate_per_capita']}])
            predicted_score = model_pipeline.predict(model_input)[0]
            
            # 6. Interpret the score into a user-friendly message package.
            message_details = interpret_score(predicted_score)

            # 7. Send the proactive alert as a push notification.
            send_push_notification(fcm_token, message_details)

        except KeyError:
            print(f"WARNING: District '{current_district}' found by geocoder but not in local crime dataset.")
        except Exception as e:
            print(f"ERROR processing AI model or message: {e}")

        # 8. Update the user's last known location in memory for the next ping.
        USER_LAST_DISTRICT[user_id] = current_district

    return jsonify({'status': 'ok'}), 200

@app.route('/end_trip', methods=['POST'])
def end_trip():
    """An endpoint for the app to call when a trip is over to clear the user's state."""
    data = request.get_json()
    user_id = data.get('user_id')
    if user_id and user_id in USER_LAST_DISTRICT:
        del USER_LAST_DISTRICT[user_id]
        print(f"Trip ended for user '{user_id}'. State cleared.")
        return jsonify({'status': 'ok', 'message': 'Trip ended successfully.'}), 200
    return jsonify({'status': 'error', 'message': 'User not found or no active trip.'}), 404


# ==============================================================================
# --- 4. SERVER EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    print("Starting Flask server for the Safety App...")
    # The server will be accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)