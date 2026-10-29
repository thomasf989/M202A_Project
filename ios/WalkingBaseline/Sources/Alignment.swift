import simd

/// Column-vector matrices. Both camera poses must share the same capture instant.
enum Alignment {
    static func mapFromSession(mapFromCamera: simd_float4x4,
                               sessionFromCamera: simd_float4x4) -> simd_float4x4 {
        mapFromCamera * simd_inverse(sessionFromCamera)
    }

    static func sessionFromMap(sessionFromMarker: simd_float4x4,
                               mapFromMarker: simd_float4x4) -> simd_float4x4 {
        sessionFromMarker * simd_inverse(mapFromMarker)
    }

    static func arCameraPose(fromCV pose: simd_float4x4) -> simd_float4x4 {
        pose * simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))
    }
}
