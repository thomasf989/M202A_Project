#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
@interface RLNativeLocalizer : NSObject
- (nullable instancetype)initWithMapPath:(NSString *)path error:(NSError **)error;
- (NSString *)metadataJSON;
- (nullable NSString *)localizeJPEG:(NSData *)jpeg cameraJSON:(NSString *)cameraJSON error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END
