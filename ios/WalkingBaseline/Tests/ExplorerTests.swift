import XCTest
import simd
@testable import WalkingBaseline

final class ExplorerTests: XCTestCase {
    func testSurveyCoverageAndRoomIsolation() throws {
        let s=WiFiSurvey(mapID:"room-a",network:"test-network",radius:1.5,samples:[
            WiFiSample(position:[0,0,0],dbm:-40,measuredAt:"2026-11-19"),
            WiFiSample(position:[2,0,0],dbm:-80,measuredAt:"2026-11-19")])
        XCTAssertEqual(try s.validated(for:"room-a").estimate(at:[0,0,0]), -40)
        XCTAssertEqual(s.estimate(at:[1,0,0]), -60)
        XCTAssertNil(s.estimate(at:[10,0,0]))
        XCTAssertThrowsError(try s.validated(for:"room-b"))
        var invalid=s;invalid.samples[0].dbm = .nan
        XCTAssertThrowsError(try invalid.validated(for:"room-a"))
    }
    func testNearestSurfacePicking() {
        let room=RoomModel(mapID:"test",name:"test",vertices:[[-1,-1,-2],[1,-1,-2],[0,1,-2],[-1,-1,-4],[1,-1,-4],[0,1,-4]],triangles:[[3,4,5],[0,1,2]])
        let hit=room.hit(origin:[0,0,0],direction:[0,0,-1])
        XCTAssertNotNil(hit);XCTAssertEqual(hit!.z,-2,accuracy:0.001)
        XCTAssertNil(room.hit(origin:[0,0,0],direction:[0,0,1]))
    }
    func testCaptureTimeAlignmentSurvivesLaterMotion() {
        var captured=matrix_identity_float4x4;captured.columns.3=[1,0,0,1]
        var mapCamera=matrix_identity_float4x4;mapCamera.columns.3=[10,2,3,1]
        let registration=Alignment.mapFromSession(mapFromCamera:mapCamera,sessionFromCamera:captured)
        var later=captured;later.columns.3.x+=2
        XCTAssertEqual((registration*later).position,SIMD3<Float>(12,2,3))
        let back=simd_inverse(registration)*(registration*later)
        XCTAssertLessThan(simd_distance(back.position,later.position),0.0001)
    }
    func testMalformedPoseRejected() {
        XCTAssertNil(simd_float4x4.fromRows([NSNumber(value:1)]))
        var values=(0..<16).map{NSNumber(value:$0 % 5 == 0 ? 1 : 0)}
        values[0]=2
        XCTAssertNil(simd_float4x4.fromRows(values))
    }
    func testPackageRejectsTruncatedData() throws {
        let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".wifiroom")
        defer {try? FileManager.default.removeItem(at:url)}
        try Data("WIFIROOM".utf8).write(to:url)
        XCTAssertThrowsError(try RoomLibrary.importPackage(url))
    }
}
