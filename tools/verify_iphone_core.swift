import Foundation
import simd

@main struct VerifyCore {
    static func main() throws {
        var checks=0
        func check(_ value: Bool,_ label: String) {precondition(value,label);checks+=1;print("PASS: \(label)")}
        let s=WiFiSurvey(mapID:"a",network:"5GHz",radius:1.5,samples:[
            WiFiSample(position:[0,0,0],dbm:-40,measuredAt:"now"),WiFiSample(position:[2,0,0],dbm:-80,measuredAt:"now")])
        check(s.estimate(at:[0,0,0]) == -40,"measurement location")
        check(s.estimate(at:[1,0,0]) == -60,"interpolation")
        check(s.estimate(at:[9,0,0]) == nil,"outside coverage")
        check((try? s.validated(for:"b")) == nil,"wrong room rejected")
        var invalid=s;invalid.samples[0].dbm = .nan
        check((try? invalid.validated(for:"a")) == nil,"nonfinite reading rejected")
        let r=RoomModel(mapID:"a",name:"A",vertices:[[-1,-1,-2],[1,-1,-2],[0,1,-2],[-1,-1,-4],[1,-1,-4],[0,1,-4]],triangles:[[3,4,5],[0,1,2]])
        check(r.hit(origin:.zero,direction:[0,0,-1])?.z == -2,"nearest mapped surface")
        check(r.hit(origin:.zero,direction:[0,0,1]) == nil,"no arbitrary tap depth")
        var captured=matrix_identity_float4x4;captured.columns.3=[1,0,0,1]
        var mapped=matrix_identity_float4x4;mapped.columns.3=[10,2,3,1]
        let registration=Alignment.mapFromSession(mapFromCamera:mapped,sessionFromCamera:captured)
        captured.columns.3.x+=2
        check((registration*captured).position == SIMD3<Float>(12,2,3),"delayed response uses capture-time pose")
        check(simd_float4x4.fromRows([1]) == nil,"invalid pose rejected")
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:folder)}
        let file=URL(fileURLWithPath:CommandLine.arguments[1])
        let imported=try RoomLibrary.importPackage(file,destinationRoot:folder)
        let room=try RoomModel.load(from:imported)
        check(!room.vertices.isEmpty,"package imported with geometry")
        check(try RoomLibrary.importPackage(file,destinationRoot:folder) == imported,"repeat import is idempotent")
        var bad=try Data(contentsOf:file);bad[bad.count-1] ^= 1
        let corrupt=folder.appendingPathComponent("corrupt.wifiroom");try bad.write(to:corrupt)
        check((try? RoomLibrary.importPackage(corrupt,destinationRoot:folder)) == nil,"checksum failure rejected")
        try Data("WIFIROOM".utf8).write(to:corrupt)
        check((try? RoomLibrary.importPackage(corrupt,destinationRoot:folder)) == nil,"truncated package rejected")
        print("\(checks) checks passed")
    }
}
