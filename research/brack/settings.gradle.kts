pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Add Liquid.ai Maven repository when available
        // maven { url = uri("https://maven.liquid.ai/releases") }
    }
}

rootProject.name = "Brack"
include(":app")