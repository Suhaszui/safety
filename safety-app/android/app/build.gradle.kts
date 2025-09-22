plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("dev.flutter.flutter-gradle-plugin")
}

// Function to read properties from local.properties file
fun readProperties(project: org.gradle.api.Project, key: String): String {
    val properties = java.util.Properties()
    val localPropertiesFile = project.rootProject.file("local.properties")
    if (localPropertiesFile.exists()) {
        properties.load(localPropertiesFile.inputStream())
    }
    return properties.getProperty(key)
}

android {
    namespace = "com.example.smart_safety_app"
    // Use the compileSdkVersion from the Flutter plugin
    compileSdk = readProperties(project, "flutter.compileSdkVersion").toInt()
    ndkVersion = readProperties(project, "flutter.ndkVersion")

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
        // Enable Core Library Desugaring
        isCoreLibraryDesugaringEnabled = true
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
    }

    defaultConfig {
        applicationId = "com.example.smart_safety_app"
        // Use the minSdkVersion from the Flutter plugin
        minSdk = readProperties(project, "flutter.minSdkVersion").toInt()
        targetSdk = readProperties(project, "flutter.targetSdkVersion").toInt()
        versionCode = readProperties(project, "flutter.versionCode").toInt()
        versionName = readProperties(project, "flutter.versionName")
        
        // Enable MultiDex
        multiDexEnabled = true
    }

    buildTypes {
        getByName("release") {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Add the dependency for Core Library Desugaring
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4")
}

