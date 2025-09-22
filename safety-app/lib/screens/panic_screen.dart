import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';

class PanicScreen extends StatefulWidget {
  const PanicScreen({super.key});

  @override
  State<PanicScreen> createState() => _PanicScreenState();
}

class _PanicScreenState extends State<PanicScreen> {
  bool isTriggered = false;
  int countdown = 5;
  Position? currentPosition;

  void startCountdown() {
    setState(() => isTriggered = true);
    Future.delayed(Duration(seconds: 1), tick);
  }

  void tick() {
    if (countdown > 1) {
      setState(() => countdown--);
      Future.delayed(Duration(seconds: 1), tick);
    } else {
      sendSOS();
    }
  }

  Future<void> sendSOS() async {
    currentPosition = await Geolocator.getCurrentPosition();
    // TODO: Send currentPosition to backend
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('SOS Sent'),
        content: Text('Your location has been shared with emergency contacts.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('OK'))
        ],
      ),
    );
  }

  void cancelSOS() {
    setState(() {
      isTriggered = false;
      countdown = 5;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.red.shade50,
      appBar: AppBar(title: Text('Emergency SOS')),
      body: Center(
        child: isTriggered
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text('Sending SOS in $countdown seconds...',
                      style: TextStyle(fontSize: 20)),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: cancelSOS,
                    style:
                        ElevatedButton.styleFrom(backgroundColor: Colors.grey),
                    child: Text('Cancel'),
                  ),
                ],
              )
            : ElevatedButton(
                onPressed: startCountdown,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  padding: EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                ),
                child: Text('SOS', style: TextStyle(fontSize: 24)),
              ),
      ),
    );
  }
}
