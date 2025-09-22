import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:geolocator/geolocator.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  late GoogleMapController mapController;
  LatLng currentLocation = LatLng(17.6868, 83.2185); // Default: Visakhapatnam

  Set<Circle> geoZones = {
    Circle(
      circleId: CircleId('green'),
      center: LatLng(17.6868, 83.2185),
      radius: 500,
      fillColor: Colors.green.withOpacity(0.3),
      strokeColor: Colors.green,
      strokeWidth: 2,
    ),
    Circle(
      circleId: CircleId('yellow'),
      center: LatLng(17.6900, 83.2200),
      radius: 300,
      fillColor: Colors.yellow.withOpacity(0.3),
      strokeColor: Colors.yellow,
      strokeWidth: 2,
    ),
    Circle(
      circleId: CircleId('red'),
      center: LatLng(17.6920, 83.2220),
      radius: 200,
      fillColor: Colors.red.withOpacity(0.3),
      strokeColor: Colors.red,
      strokeWidth: 2,
    ),
  };

  @override
  void initState() {
    super.initState();
    _getLocation();
  }

  Future<void> _getLocation() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.requestPermission();
    if (permission == LocationPermission.denied) return;

    Position position = await Geolocator.getCurrentPosition();
    setState(() {
      currentLocation = LatLng(position.latitude, position.longitude);
    });

    _checkZone(position.latitude, position.longitude);
  }

  void _checkZone(double lat, double lng) {
    // Simple zone check (replace with backend call if needed)
    double distanceToRed =
        Geolocator.distanceBetween(lat, lng, 17.6920, 83.2220);
    if (distanceToRed < 200) {
      _showAlert('⚠️ You entered a high-risk zone!');
    }
  }

  void _showAlert(String message) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Geo-Fence Alert'),
        content: Text(message),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('OK'))
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Live Map')),
      body: GoogleMap(
        initialCameraPosition:
            CameraPosition(target: currentLocation, zoom: 15),
        circles: geoZones,
        myLocationEnabled: true,
        onMapCreated: (controller) => mapController = controller,
      ),
    );
  }
}
