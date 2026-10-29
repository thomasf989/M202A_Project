import XCTest
import simd
@testable import WalkingBaseline

final class AlignmentTests: XCTestCase {
    func testMarkerRegistrationAndCurrentCamera() {
        var mapFromMarker = matrix_identity_float4x4
        mapFromMarker.columns.3 = [2, 0, 0, 1]
        let rotation = simd_float4x4(simd_quatf(angle: .pi / 2, axis: [0, 1, 0]))
        var sessionFromMarker = rotation
        sessionFromMarker.columns.3 = [1, 0, 3, 1]
        let registration = Alignment.sessionFromMap(
            sessionFromMarker: sessionFromMarker, mapFromMarker: mapFromMarker)
        let mappedMarker = registration * mapFromMarker
        for column in 0..<4 {
            XCTAssertLessThan(simd_length(mappedMarker[column] - sessionFromMarker[column]), 0.0001)
        }
        let mapFromSession = Alignment.mapFromSession(
            mapFromCamera: mapFromMarker, sessionFromCamera: sessionFromMarker)
        let identity = registration * mapFromSession
        for column in 0..<4 {
            XCTAssertLessThan(simd_length(identity[column] - matrix_identity_float4x4[column]), 0.0001)
        }
    }

    func testCVForwardBecomesARNegativeZ() {
        let converted = Alignment.arCameraPose(fromCV: matrix_identity_float4x4)
        XCTAssertEqual(converted * SIMD4<Float>(0, 0, -1, 0), SIMD4<Float>(0, 0, 1, 0))
    }
}
