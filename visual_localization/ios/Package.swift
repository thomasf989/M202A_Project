// swift-tools-version:5.9
import PackageDescription

// Portable archive tests, independent of the UIKit app and OpenCV package.
let package = Package(
    name: "RoomArchiveChecks",
    platforms: [.macOS(.v12), .iOS(.v16)],
    dependencies: [.package(url: "https://github.com/weichsel/ZIPFoundation.git", exact: "0.9.20")],
    targets: [
        .target(name: "RoomArchive", dependencies: ["ZIPFoundation"], path: "RoomLocate",
                exclude: ["AppModel.swift", "ContentView.swift", "Info.plist", "LocalRoomInput.swift", "Models.swift",
                          "NativeLocalizer.h", "NativeLocalizer.mm", "NewRoomView.swift", "PhotoInput.swift",
                          "RoomLocate-Bridging-Header.h", "RoomLocateApp.swift", "RoomScene.swift", "RoomStore.swift", "ThirdPartyNotices.txt"],
                sources: ["ReferenceArchive.swift"]),
        .testTarget(name: "RoomArchiveTests", dependencies: ["RoomArchive", "ZIPFoundation"], path: "tests/swift")
    ]
)
