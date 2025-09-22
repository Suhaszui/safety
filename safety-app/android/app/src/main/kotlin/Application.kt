package com.example.smart_safety_app // Make sure this matches your package name

import io.flutter.app.FlutterApplication
import io.flutter.plugin.common.PluginRegistry
import io.flutter.plugin.common.PluginRegistry.PluginRegistrantCallback
import rekab.app.background_locator_2.BackgroundLocator2Plugin

class Application : FlutterApplication(), PluginRegistrantCallback {
    override fun registerWith(registry: PluginRegistry?) {
        if (registry != null) {
            BackgroundLocator2Plugin.registerWith(registry)
        }
    }
}