import 'package:flutter/material.dart';

class ItineraryScreen extends StatefulWidget {
  const ItineraryScreen({super.key});

  @override
  State<ItineraryScreen> createState() => _ItineraryScreenState();
}

class _ItineraryScreenState extends State<ItineraryScreen> {
  final List<Map<String, String>> itinerary = [];

  final TextEditingController locationController = TextEditingController();
  final TextEditingController timeController = TextEditingController();

  void addDestination() {
    final location = locationController.text.trim();
    final time = timeController.text.trim();
    if (location.isEmpty || time.isEmpty) return;

    setState(() {
      itinerary.add({'location': location, 'time': time});
      locationController.clear();
      timeController.clear();
    });
  }

  void markCheckIn(int index) {
    setState(() {
      itinerary[index]['status'] = 'Checked In';
    });
  }

  void markCheckOut(int index) {
    setState(() {
      itinerary[index]['status'] = 'Checked Out';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Trip Itinerary')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            TextField(
              controller: locationController,
              decoration: InputDecoration(labelText: 'Destination'),
            ),
            TextField(
              controller: timeController,
              decoration:
                  InputDecoration(labelText: 'Time (e.g. 2025-09-21 14:00)'),
            ),
            SizedBox(height: 10),
            ElevatedButton(
                onPressed: addDestination, child: Text('Add Destination')),
            SizedBox(height: 20),
            Expanded(
              child: ListView.builder(
                itemCount: itinerary.length,
                itemBuilder: (context, index) {
                  final item = itinerary[index];
                  return Card(
                    child: ListTile(
                      title: Text(item['location'] ?? ''),
                      subtitle: Text(
                          '${item['time']} • ${item['status'] ?? 'Pending'}'),
                      trailing: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            icon: Icon(Icons.check),
                            onPressed: () => markCheckIn(index),
                          ),
                          IconButton(
                            icon: Icon(Icons.exit_to_app),
                            onPressed: () => markCheckOut(index),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
