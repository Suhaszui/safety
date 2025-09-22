import 'package:flutter/material.dart';

class EndTripScreen extends StatelessWidget {
  const EndTripScreen({super.key});

  void confirmEndTrip(BuildContext context) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('End Trip'),
        content: Text('Are you sure you want to end your trip and logout?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          TextButton(
            onPressed: () {
              // TODO: Clear session and logout
              Navigator.popUntil(context, ModalRoute.withName('/login'));
            },
            child: Text('Confirm'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('End Trip')),
      body: Center(
        child: ElevatedButton(
          onPressed: () => confirmEndTrip(context),
          style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
          child: Text('End Trip & Logout'),
        ),
      ),
    );
  }
}
