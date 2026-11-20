#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@interface OfflineLocalizer : NSObject
- (nullable instancetype)initWithDatabase:(NSString *)path error:(NSError **)error;
// Input is a sensor-oriented JPEG and its row-major intrinsic matrix.
- (NSDictionary *)localizeJPEG:(NSData *)jpeg intrinsics:(NSArray<NSNumber *> *)intrinsics;
@end
NS_ASSUME_NONNULL_END
