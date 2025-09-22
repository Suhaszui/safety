import 'package:flutter/material.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool trackingEnabled = true;
  bool notificationsEnabled = true;
  String selectedLanguage = 'English';

  final List<String> languages = [
    'English',
    'Hindi',
    'Telugu',
    'Tamil',
    'Bengali',
    'Kannada',
    'Malayalam',
    'Gujarati',
    'Marathi',
    'Punjabi',
    'Urdu'
  ];

  final List<String> emergencyContacts = ['+91 9876543210', '+91 9123456780'];

  void toggleTracking(bool value) {
    setState(() => trackingEnabled = value);
  }

  void toggleNotifications(bool value) {
    setState(() => notificationsEnabled = value);
  }

  void updateLanguage(String? value) {
    if (value != null) setState(() => selectedLanguage = value);
  }

  void removeContact(int index) {
    setState(() => emergencyContacts.removeAt(index));
  }

  void addContact() {
    // TODO: Add contact picker or input
    setState(() => emergencyContacts.add('+91 9000000000'));
  }

  void wipeData() {
    // TODO: Clear local storage and logout
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Wipe Data'),
        content: Text('All app data will be deleted. Continue?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          TextButton(
              onPressed: () {
                // TODO: Perform wipe
                Navigator.pop(context);
              },
              child: Text('Confirm')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: ListView(
        padding: EdgeInsets.all(20),
        children: [
          Text('Language', style: TextStyle(fontWeight: FontWeight.bold)),
          DropdownButton<String>(
            value: selectedLanguage,
            items: languages.map((lang) {
              return DropdownMenuItem(value: lang, child: Text(lang));
            }).toList(),
            onChanged: updateLanguage,
          ),
          Divider(),
          SwitchListTile(
            title: Text('Enable Real-Time Tracking'),
            value: trackingEnabled,
            onChanged: toggleTracking,
          ),
          SwitchListTile(
            title: Text('Enable Notifications'),
            value: notificationsEnabled,
            onChanged: toggleNotifications,
          ),
          Divider(),
          Text('Emergency Contacts',
              style: TextStyle(fontWeight: FontWeight.bold)),
          ...emergencyContacts.map((contact) {
            int index = emergencyContacts.indexOf(contact);
            return ListTile(
              title: Text(contact),
              trailing: IconButton(
                icon: Icon(Icons.delete),
                onPressed: () => removeContact(index),
              ),
            );
          }).toList(),
          TextButton(onPressed: addContact, child: Text('Add Contact')),
          Divider(),
          ListTile(
            title: Text('Logout & Wipe Data'),
            trailing: Icon(Icons.delete_forever),
            onTap: wipeData,
          ),
        ],
      ),
    );
  }
}
