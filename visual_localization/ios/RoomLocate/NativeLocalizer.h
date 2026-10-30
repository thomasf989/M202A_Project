#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
@interface RLNativeLocalizer : NSObject
+ (nullable NSString *)buildRoomWithGLBPath:(NSString *)glbPath
                            referenceRoot:(NSString *)referenceRoot
                          outputDirectory:(NSString *)outputDirectory
                                     name:(NSString *)name
                            maxReferences:(NSInteger)maxReferences
                                 progress:(BOOL (^)(double fraction, NSString *message))progress
                                    error:(NSError **)error
    NS_SWIFT_NAME(buildRoom(glbPath:referenceRoot:outputDirectory:name:maxReferences:progress:));
- (nullable instancetype)initWithMapPath:(NSString *)path error:(NSError **)error;
- (NSString *)metadataJSON;
- (nullable NSString *)localizeJPEG:(NSData *)jpeg cameraJSON:(NSString *)cameraJSON error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END
