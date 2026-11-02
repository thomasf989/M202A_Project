import SwiftUI

@main
@MainActor
struct RoomLocateApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene {
        WindowGroup {
            ContentView(model: model).onOpenURL { url in Task { await model.importRoom(url) } }
        }
    }
}
