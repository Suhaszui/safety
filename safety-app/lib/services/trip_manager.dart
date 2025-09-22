import 'package:permission_handler/permission_handler.dart';
import 'package:background_locator_2/background_locator.dart';
import 'package:background_locator_2/settings/android_settings.dart';
import 'package:background_locator_2/settings/ios_settings.dart';

// **** THIS IS THE MISSING LINE ****
// This import makes the LocationAccuracy enum available.
import 'package:background_locator_2/settings/locator_settings.dart';

import 'location_callback_handler.dart';

class TripManager {
  Future<void> _requestLocationPermission() async {
    final status = await Permission.location.request();
    if (status.isGranted) {
      await Permission.locationAlways.request();
    }
  }

  void startTrip() async {
    await _requestLocationPermission();
    await BackgroundLocator.initialize();

    if (await BackgroundLocator.isServiceRunning()) {
      print("Trip service is already running.");
      return;
    }

    await BackgroundLocator.registerLocationUpdate(callback,
        androidSettings: AndroidSettings(
          accuracy: LocationAccuracy.HIGH, // This will now be found
          interval: 60,
          distanceFilter: 100,

          androidNotificationSettings: AndroidNotificationSettings(
            notificationChannelName: 'Location Tracking',
            notificationTitle: "Trip in Progress",
            notificationMsg: "Actively monitoring your safety.",
            notificationIcon: 'mipmap/ic_launcher',
          ),
        ),
        iosSettings: IOSSettings(
          accuracy: LocationAccuracy.HIGH, // This will now be found
          distanceFilter: 100,
          showsBackgroundLocationIndicator: true,
        ));
    print("Trip started successfully!");
  }

  void stopTrip() {
    BackgroundLocator.unRegisterLocationUpdate();
    print("Trip stopped.");
  }
}
