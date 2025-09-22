import 'package:flutter/material.dart';
import '../services/trip_service.dart';

class TripListScreen extends StatefulWidget {
  @override
  _TripListScreenState createState() => _TripListScreenState();
}

class _TripListScreenState extends State<TripListScreen> {
  final TripService _tripService = TripService.instance;
  late Future<List<Map<String, dynamic>>> _trips;

  @override
  void initState() {
    super.initState();
    _trips = _tripService.getTrips(
        userId: 0); // Replace 0 with the actual user ID if available
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Your Trips')),
      body: FutureBuilder<List<Map<String, dynamic>>>(
        future: _trips,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No trips found.'));
          }

          final trips = snapshot.data!;
          return ListView.builder(
            itemCount: trips.length,
            itemBuilder: (context, index) {
              final trip = trips[index];
              return Card(
                margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: ListTile(
                  title: Text(trip['location']),
                  subtitle: Text('Status: ${trip['status']}'),
                  trailing: Text(
                    trip['time'].toString().substring(0, 16),
                    style: TextStyle(fontSize: 12),
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
