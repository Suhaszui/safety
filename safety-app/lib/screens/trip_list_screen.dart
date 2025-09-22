import 'package:flutter/material.dart';
import '../services/trip_service.dart';

class ItineraryScreen extends StatefulWidget {
  @override
  _ItineraryScreenState createState() => _ItineraryScreenState();
}

class _ItineraryScreenState extends State<ItineraryScreen> {
  final TripService _tripService = TripService.instance;
  late Future<List<Map<String, dynamic>>> _trips;
  final int userId = 1; // Replace with actual logged-in user ID

  @override
  void initState() {
    super.initState();
    _loadTrips();
  }

  void _loadTrips() {
    setState(() {
      _trips = _tripService.getTrips(userId: userId);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Upcoming Trips')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: () async {
                await Navigator.pushNamed(context, '/add_trip');
                _loadTrips(); // Refresh after adding
              },
              child: Text('Add Your Trip Details'),
            ),
          ),
          Expanded(
            child: FutureBuilder<List<Map<String, dynamic>>>(
              future: _trips,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(child: Text('Error: ${snapshot.error}'));
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return Center(child: Text('No upcoming trips.'));
                }

                final trips = snapshot.data!;
                return ListView.builder(
                  itemCount: trips.length,
                  itemBuilder: (context, index) {
                    final trip = trips[index];
                    return ListTile(
                      title: Text(trip['location']),
                      subtitle: Text('Status: ${trip['status']}'),
                      trailing: Text(trip['time'].toString().substring(0, 16)),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
