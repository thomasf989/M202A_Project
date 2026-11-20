import Foundation
import CryptoKit

struct RoomChoice: Identifiable {
    let id: String
    let name: String
    let folder: URL
}
enum RoomLibrary {
    private struct Summary: Codable { let mapID: String; let name: String }
    static var root: URL {
        FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Rooms")
    }
    static func choices() -> [RoomChoice] {
        var result: [RoomChoice]=[]
        var folders=(try? FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)) ?? []
        if let bundled=Bundle.main.url(forResource:"Room",withExtension:nil) {folders.insert(bundled,at:0)}
        for folder in folders {
            if let data=try? Data(contentsOf:folder.appendingPathComponent("summary.json")),
               let room=try? JSONDecoder().decode(Summary.self,from:data), !result.contains(where:{$0.id==room.mapID}) {
                result.append(RoomChoice(id:room.mapID,name:room.name,folder:folder))
            }
        }
        return result
    }
    private struct Package: Decodable {
        struct File: Decodable {let name: String;let size: Int;let sha256: String}
        let version: Int;let mapID: String;let name: String;let files: [File]
    }
    static func importPackage(_ url: URL, destinationRoot: URL = RoomLibrary.root) throws -> URL {
        let access=url.startAccessingSecurityScopedResource(); defer {if access {url.stopAccessingSecurityScopedResource()}}
        let size=try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
        guard size>12 && size<600_000_000 else {throw RoomError.invalidRoom}
        let file=try FileHandle(forReadingFrom:url); defer {try? file.close()}
        guard try file.read(upToCount:8)==Data("WIFIROOM".utf8), let bytes=try file.read(upToCount:4),bytes.count==4 else {throw RoomError.invalidRoom}
        let count=bytes.enumerated().reduce(0){$0 | (Int($1.element) << ($1.offset*8))}
        guard count>0 && count<65536,let header=try file.read(upToCount:count),header.count==count else {throw RoomError.invalidRoom}
        let p=try JSONDecoder().decode(Package.self,from:header)
        guard p.version==1,p.mapID.count==64,p.mapID.allSatisfy({$0.isHexDigit}),
              p.files.map(\.name)==["room.json","features.json"],
              p.files.allSatisfy({$0.size>0 && $0.size<500_000_000}),
              p.files.reduce(12+count,{$0+$1.size})==size else {throw RoomError.invalidRoom}
        try FileManager.default.createDirectory(at:destinationRoot,withIntermediateDirectories:true)
        let stage=destinationRoot.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:stage,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:stage)}
        for item in p.files {
            let target=stage.appendingPathComponent(item.name)
            FileManager.default.createFile(atPath:target.path,contents:nil)
            let out=try FileHandle(forWritingTo:target); defer {try? out.close()}
            var remaining=item.size;var hash=SHA256()
            while remaining>0 {
                guard let chunk=try file.read(upToCount:min(remaining,1_048_576)),!chunk.isEmpty else {throw RoomError.invalidRoom}
                try out.write(contentsOf:chunk);hash.update(data:chunk);remaining-=chunk.count
            }
            let digest=hash.finalize().map{String(format:"%02x",$0)}.joined()
            guard digest==item.sha256 else {throw RoomError.invalidRoom}
        }
        let room=try RoomModel.load(from:stage)
        guard room.mapID==p.mapID else {throw RoomError.invalidRoom}
        try JSONEncoder().encode(Summary(mapID:room.mapID,name:room.name)).write(to:stage.appendingPathComponent("summary.json"))
        let destination=destinationRoot.appendingPathComponent(p.mapID,isDirectory:true)
        if FileManager.default.fileExists(atPath:destination.path) {return destination}
        try FileManager.default.moveItem(at:stage,to:destination)
        return destination
    }
}
