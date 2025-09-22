import 'package:flutter/material.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final Map<String, String> profile = {
      'Name': 'Manisha Reddy',
      'Nationality': 'Indian',
      'Passport No': 'M1234567',
      'Tourist Code': 'TS-0921-IND',
      'Trip Validity': 'Sep 30, 2025',
    };

    return Scaffold(
      appBar: AppBar(title: Text('Digital ID')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            CircleAvatar(
                radius: 50, backgroundImage: AssetImage('assets/profile.jpg')),
            SizedBox(height: 20),
            ...profile.entries.map((entry) => ListTile(
                  title: Text(entry.key),
                  subtitle: Text(entry.value),
                )),
            Divider(),
            ElevatedButton(
              onPressed: () {
                // TODO: Export or share ID
              },
              child: Text('Share Digital ID'),
            ),
          ],
        ),
      ),
    );
  }
}
