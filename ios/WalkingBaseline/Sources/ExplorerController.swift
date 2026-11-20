import SwiftUI
import ARKit
import RealityKit
import CoreImage

/// Delegate queue and all UI/AR mutations run on main; inference runs on its own serial queue.
final class ExplorerController: NSObject, ObservableObject, ARSessionDelegate {
    let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Loading offline room…"
    @Published var reading = "No Wi-Fi measurements imported"
    @Published var selected = "Tap a mapped surface to inspect Wi-Fi"
    @Published var poseText = "Location not established"
    @Published var busy = false
    @Published var ready = false
    @Published var aligned = false
    @Published var sampleCount = 0
    @Published var network = "Not surveyed"
    private(set) var room: RoomModel?
    private(set) var survey: WiFiSurvey?
    private var mapFromSession: simd_float4x4?
    private var root: AnchorEntity?
    private var sessionID = UUID()
    private var roomLoadID = UUID()
    private var trackingNormal = false
    private var lastUpdate: TimeInterval = 0
    private var lastAlignment: TimeInterval = 0
    private let worker = DispatchQueue(label: "offline.localizer", qos: .userInitiated)
    private var localizer: OfflineLocalizer? // only read/written on worker
    private let ciContext = CIContext()

    @Published var rooms: [RoomChoice] = []
    @Published var roomName = "No room selected"
    override init() {
        super.init()
        rooms = RoomLibrary.choices()
        let saved = UserDefaults.standard.string(forKey: "selectedRoom")
        if let first = rooms.first(where: {$0.id == saved}) ?? rooms.first { selectRoom(first) }
        else { status = "Import an offline .wifiroom package in Survey tools." }
    }
    private var surveyURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("wifi-\(room?.mapID ?? "none").json")
    }
    func selectRoom(_ choice: RoomChoice) {
        invalidate(); ready=false; room=nil; survey=nil; sampleCount=0; network="Not surveyed"
        roomName=choice.name; status="Loading offline room…"
        roomLoadID=UUID(); let epoch=roomLoadID
        worker.async { [weak self] in
            guard let self else {return}
            do {
                let model=try RoomModel.load(from:choice.folder)
                self.localizer=nil
                self.localizer=try OfflineLocalizer(database:choice.folder.appendingPathComponent("features.json").path)
                DispatchQueue.main.async {
                    guard self.roomLoadID==epoch else {return}
                    self.room=model
                    self.survey=WiFiSurvey(mapID:model.mapID,network:"Not surveyed",radius:1.5,samples:[])
                    if let data=try? Data(contentsOf:self.surveyURL),
                       let saved=try? JSONDecoder().decode(WiFiSurvey.self,from:data).validated(for:model.mapID) {self.survey=saved}
                    self.ready=true;self.updateSurveyLabels()
                    UserDefaults.standard.set(model.mapID,forKey:"selectedRoom")
                    self.status="Room ready. Move slowly, then Locate me."
                }
            } catch {DispatchQueue.main.async {if self.roomLoadID==epoch {self.status=error.localizedDescription}}}
        }
    }
    func importRoom(_ url: URL) {
        guard !busy else {return};busy=true
        worker.async { [weak self] in
            guard let self else {return}
            do {
                let folder=try RoomLibrary.importPackage(url)
                let choices=RoomLibrary.choices()
                DispatchQueue.main.async {
                    self.busy=false;self.rooms=choices
                    if let choice=choices.first(where:{$0.folder.path==folder.path}) ?? choices.last {self.selectRoom(choice)}
                }
            } catch {DispatchQueue.main.async {self.busy=false;self.status=error.localizedDescription}}
        }
    }
    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            status = "Simulator: UI only. Camera localization and walking require a physical iPhone."
            return
        }
        view.session.delegate = self
        view.session.delegateQueue = .main
        let gesture = UITapGestureRecognizer(target: self, action: #selector(tapped(_:)))
        view.addGestureRecognizer(gesture)
        reset()
    }
    func reset() {
        invalidate()
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        view.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        status = "Move slowly until tracking is normal, then Locate me."
    }
    private func invalidate() {
        sessionID = UUID(); mapFromSession = nil; aligned = false; trackingNormal = false
        root?.removeFromParent(); root = nil
        poseText = "Location not established"; reading = "Locate first to read Wi-Fi"
        selected = "Tap a mapped surface after localization"
    }
    func suspend() { invalidate(); view.session.pause() }
    func locate() {
        guard ready, !busy, trackingNormal, let frame=view.session.currentFrame else {
            status = "Wait for normal tracking on a physical iPhone."; return
        }
        // Preserve capture-time camera pose, intrinsics, and unrotated sensor pixels together.
        let epoch=sessionID, capturedPose=frame.camera.transform, capturedTime=frame.timestamp
        let k=frame.camera.intrinsics, buffer=frame.capturedImage
        let width=CVPixelBufferGetWidth(buffer), height=CVPixelBufferGetHeight(buffer)
        let scale=min(1.0, 1280.0/Double(max(width,height)))
        let outputWidth=Int(Double(width)*scale), outputHeight=Int(Double(height)*scale)
        let sx=Double(outputWidth)/Double(width), sy=Double(outputHeight)/Double(height)
        let intrinsics: [NSNumber] = [NSNumber(value: Double(k[0].x)*sx),0,NSNumber(value: Double(k[2].x)*sx),
                                    0,NSNumber(value: Double(k[1].y)*sy),NSNumber(value: Double(k[2].y)*sy),0,0,1]
        busy=true; status="Locating on this iPhone…"
        worker.async { [weak self] in
            guard let self else { return }
            let image=CIImage(cvPixelBuffer: buffer).transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            let result: [AnyHashable: Any]
            if let jpeg=self.ciContext.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]),
               let localizer=self.localizer {
                result=localizer.localizeJPEG(jpeg, intrinsics: intrinsics)
            } else { result=["success": false, "reason": "Could not capture camera image."] }
            DispatchQueue.main.async {
                self.busy=false
                guard self.sessionID==epoch, self.trackingNormal,
                      let current=self.view.session.currentFrame, current.timestamp-capturedTime<120 else {
                    self.status="Tracking/session changed during localization. Please locate again."; return
                }
                guard result["success"] as? Bool == true, let rows=result["pose"] as? [NSNumber],
                      let mapFromCV=simd_float4x4.fromRows(rows) else {
                    self.status=result["reason"] as? String ?? "Pose rejected. Try another view."
                    return
                }
                let mapFromCamera=Alignment.arCameraPose(fromCV: mapFromCV)
                self.mapFromSession=Alignment.mapFromSession(mapFromCamera: mapFromCamera, sessionFromCamera: capturedPose)
                self.aligned=true; self.lastAlignment=current.timestamp
                self.rebuildOverlay()
                let photoSignal=self.survey?.estimate(at: mapFromCamera.position)
                self.selected="Photo location: " + self.describe(photoSignal)
                self.status="Located offline (\(result["inliers"] ?? 0) inliers). Walk and turn; tap a surface."
            }
        }
    }
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let normal: Bool
        if case .normal=frame.camera.trackingState {normal=true} else {normal=false}
        if trackingNormal && !normal {invalidate();status="Tracking lost. Locate again when tracking is normal."}
        trackingNormal=normal
        root?.isEnabled=trackingNormal
        if !trackingNormal { reading="Tracking limited — readings hidden"; selected="Tracking limited — locate again if drifted"; return }
        guard frame.timestamp-lastUpdate>0.2 else { return }; lastUpdate=frame.timestamp
        guard let mapFromSession else {
            if ready && !busy { status="Offline room ready. Tap Locate me." }; return
        }
        let pose=mapFromSession*frame.camera.transform, p=pose.position
        let forward = -SIMD3(pose.columns.2.x, pose.columns.2.y, pose.columns.2.z)
        let yaw=atan2(forward.x,-forward.z)*180 / .pi
        poseText=String(format:"x %.2f  y %.2f  z %.2f m · yaw %.0f°",p.x,p.y,p.z,yaw)
        reading="Here: " + describe(survey?.estimate(at:p))
        if frame.timestamp-lastAlignment>60 && !busy { status="AR tracking can drift. Tap Locate me to refresh alignment." }
    }
    func sessionWasInterrupted(_ session: ARSession) { invalidate(); status="Session interrupted. Locate again after returning." }
    func sessionInterruptionEnded(_ session: ARSession) { reset() }
    func session(_ session: ARSession, didFailWithError error: Error) { invalidate(); status=error.localizedDescription }
    private func describe(_ signal: Float?) -> String {
        signal.map { String(format:"≈ %.0f dBm (survey estimate)", $0) } ?? "No measured coverage"
    }
    @objc private func tapped(_ recognizer: UITapGestureRecognizer) {
        guard trackingNormal, let transform=mapFromSession, let room,
              let ray=view.ray(through: recognizer.location(in:view)) else { return }
        let origin=(transform*SIMD4(ray.origin,1)).xyz
        let direction=simd_normalize((transform*SIMD4(ray.direction,0)).xyz)
        guard let p=room.hit(origin:origin,direction:direction) else { selected="No mapped surface at this point"; return }
        selected=String(format:"Tapped (%.2f, %.2f, %.2f m): ",p.x,p.y,p.z)+describe(survey?.estimate(at:p))
        if let root {
            root.children.filter { $0.name=="selection" }.forEach { $0.removeFromParent() }
            let dot=ModelEntity(mesh:.generateSphere(radius:0.035),materials:[UnlitMaterial(color:.white)])
            dot.name="selection"; dot.position=p; root.addChild(dot)
        }
    }
    func importSurvey(_ url: URL) {
        let access=url.startAccessingSecurityScopedResource(); defer { if access {url.stopAccessingSecurityScopedResource()} }
        do {
            guard let room else {throw RoomError.invalidRoom}
            let size=try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
            guard size<2_000_000 else {throw RoomError.invalidSurvey}
            let next=try JSONDecoder().decode(WiFiSurvey.self,from:Data(contentsOf:url)).validated(for:room.mapID)
            try JSONEncoder().encode(next).write(to:surveyURL,options:.atomic)
            survey=next; updateSurveyLabels(); rebuildOverlay(); status="Survey imported for \(next.network)."
        } catch {status=error.localizedDescription}
    }
    func addMeasurement(dbm: Float, network: String) {
        guard trackingNormal, let transform=mapFromSession, let frame=view.session.currentFrame, var s=survey else {status="Locate first with normal tracking.";return}
        guard dbm.isFinite, (-120 ... -10).contains(dbm), !network.trimmingCharacters(in:.whitespaces).isEmpty else {status="Enter a network name and a dBm reading from −120 to −10.";return}
        guard s.samples.isEmpty || s.network==network else {status="Import an empty survey before changing networks; readings must not be mixed.";return}
        guard s.samples.count<5000 else {status="Survey limit reached.";return}
        let p=(transform*frame.camera.transform).position
        s.network=network; s.samples.append(WiFiSample(position:[p.x,p.y,p.z],dbm:dbm,measuredAt:ISO8601DateFormatter().string(from:Date())))
        do {try JSONEncoder().encode(s).write(to:surveyURL,options:.atomic);survey=s;updateSurveyLabels();rebuildOverlay();status="External measurement recorded at current phone position."}
        catch {status=error.localizedDescription}
    }
    func exportSurvey() -> URL? {
        guard let survey else { return nil }
        do {try JSONEncoder().encode(survey).write(to:surveyURL,options:.atomic);return surveyURL}
        catch {status=error.localizedDescription;return nil}
    }
    private func updateSurveyLabels() {sampleCount=survey?.samples.count ?? 0;network=survey?.network ?? "Not surveyed"}
    private func rebuildOverlay() {
        root?.removeFromParent();root=nil
        guard let transform=mapFromSession, let room, let survey else {return}
        let anchor=AnchorEntity(world:simd_inverse(transform)); let points=room.points
        // Depth-only scanned geometry hides heat behind nearer mapped surfaces.
        var descriptor=MeshDescriptor(name:"scanned-room-occlusion")
        descriptor.positions=MeshBuffers.Positions(points)
        descriptor.primitives = .triangles(room.triangles.flatMap {$0.map {UInt32($0)}})
        if let mesh=try? MeshResource.generate(from:[descriptor]) {
            anchor.addChild(ModelEntity(mesh:mesh,materials:[OcclusionMaterial()]))
        }
        let step=max(1,room.triangles.count/1200)
        // Surface dots only within measured 3D coverage. Empty survey renders no invented heat.
        for i in stride(from:0,to:room.triangles.count,by:step) {
            let t=room.triangles[i], p=(points[t[0]]+points[t[1]]+points[t[2]])/3
            guard let value=survey.estimate(at:p) else {continue}
            let strength=max(0,min(1,(value+90)/50))
            let color=UIColor(hue:CGFloat(strength/3),saturation:0.85,brightness:0.95,alpha:0.7)
            let dot=ModelEntity(mesh:.generateSphere(radius:0.045),materials:[UnlitMaterial(color:color)])
            dot.position=p;anchor.addChild(dot)
        }
        for sample in survey.samples {
            let dot=ModelEntity(mesh:.generateSphere(radius:0.025),materials:[UnlitMaterial(color:.cyan)])
            dot.position=SIMD3(sample.position[0],sample.position[1],sample.position[2]);anchor.addChild(dot)
        }
        view.scene.addAnchor(anchor);root=anchor;anchor.isEnabled=trackingNormal
    }
}
private extension SIMD4 where Scalar==Float {var xyz: SIMD3<Float> {SIMD3(x,y,z)}}
