// Include C++ headers before UIKit, whose YES/NO macros collide with OpenCV.
#include "../Native/LocalizerCore.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/ios.h>
#include <memory>
#include <stdexcept>
#import "NativeLocalizer.h"
#import <UIKit/UIKit.h>

static NSError *RoomError(const std::exception& e) {
    return [NSError errorWithDomain:@"RoomLocate" code:1 userInfo:@{
        NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()] ?: @"定位处理失败"
    }];
}

@implementation RLNativeLocalizer {
    std::unique_ptr<roomloc::Localizer> _engine;
}
- (nullable instancetype)initWithMapPath:(NSString *)path error:(NSError **)error {
    self = [super init];
    if (self) {
        try { _engine = std::make_unique<roomloc::Localizer>(path.fileSystemRepresentation); }
        catch (const std::exception& e) { if (error) *error = RoomError(e); return nil; }
    }
    return self;
}
- (NSString *)metadataJSON {
    return [NSString stringWithUTF8String:_engine->metadataJSON().c_str()];
}
- (nullable NSString *)localizeJPEG:(NSData *)jpeg cameraJSON:(NSString *)cameraJSON error:(NSError **)error {
    try {
        UIImage *image = [UIImage imageWithData:jpeg];
        if (!image) throw std::runtime_error("无法读取照片");
        cv::Mat rgba, bgr;
        UIImageToMat(image, rgba, true);
        if (rgba.channels() == 4) cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        else if (rgba.channels() == 3) cv::cvtColor(rgba, bgr, cv::COLOR_RGB2BGR);
        else bgr = rgba;
        auto json = _engine->localize(bgr, cameraJSON.UTF8String);
        return [NSString stringWithUTF8String:json.c_str()];
    } catch (const std::exception& e) { if (error) *error = RoomError(e); return nil; }
}
@end
