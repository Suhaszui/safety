import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/trip_service.dart';

class AddTripScreen extends StatefulWidget {
  @override
  _AddTripScreenState createState() => _AddTripScreenState();
}

class _AddTripScreenState extends State<AddTripScreen> {
  final TripService _tripService = TripService.instance;
  final TextEditingController locationController = TextEditingController();
  final TextEditingController timeController = TextEditingController();
  final TextEditingController statusController = TextEditingController();

  Future<void> submitTrip() async {
    final prefs = await SharedPreferences.getInstance();
    final userId = prefs.getInt('user_id');

    if (userId == null) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('User not logged in')));
      return;
    }

    try {
      DateTime.parse(timeController.text);
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Invalid time format')));
      return;
    }

    final success = await _tripService.addTrip({
      'user_id': userId,
      'location': locationController.text,
      'time': timeController.text,
      'status': statusController.text,
    });

    if (success) {
      Navigator.pop(context);
    } else {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Failed to add trip')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Add New Trip')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
                controller: locationController,
                decoration: InputDecoration(labelText: 'Location')),
            TextField(
                controller: timeController,
                decoration:
                    InputDecoration(labelText: 'Time (YYYY-MM-DD HH:MM:SS)')),
            TextField(
                controller: statusController,
                decoration: InputDecoration(labelText: 'Status')),
            SizedBox(height: 20),
            ElevatedButton(onPressed: submitTrip, child: Text('Submit Trip')),
          ],
        ),
      ),
    );
  }
}
