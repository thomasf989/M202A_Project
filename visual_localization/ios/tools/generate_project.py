#!/usr/bin/env python3
"""Regenerate the checked-in, dependency-pinned Xcode project (stdlib only)."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def identifier(value):
    return hashlib.sha1(value.encode()).hexdigest()[:24].upper()


class Ref(str):
    pass


def encode(value, level=0):
    if isinstance(value, Ref):
        return str(value)
    if isinstance(value, dict):
        return '{\n' + ''.join('\t' * (level + 1) + json.dumps(k) + ' = ' + encode(v, level + 1) + ';\n' for k, v in value.items()) + '\t' * level + '}'
    if isinstance(value, list):
        return '(' + ', '.join(encode(v, level) for v in value) + ')'
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def main():
    objects = {}
    def add(key, isa, **fields):
        ref = Ref(identifier(key)); objects[str(ref)] = dict(isa=isa, **fields); return ref
    product = add('app-product', 'PBXFileReference', explicitFileType='wrapper.application', path='RoomLocate.app', sourceTree='BUILT_PRODUCTS_DIR', includeInIndex=0)
    package = add('opencv-package', 'XCRemoteSwiftPackageReference', repositoryURL='https://github.com/yeatse/opencv-spm.git', requirement=dict(kind='exactVersion', version='4.12.0'))
    opencv = add('opencv-product', 'XCSwiftPackageProductDependency', package=package, productName='OpenCV')
    framework = add('opencv-link', 'PBXBuildFile', productRef=opencv)
    zip_package = add('zip-package', 'XCRemoteSwiftPackageReference', repositoryURL='https://github.com/weichsel/ZIPFoundation.git', requirement=dict(kind='exactVersion', version='0.9.20'))
    zip_product = add('zip-product', 'XCSwiftPackageProductDependency', package=zip_package, productName='ZIPFoundation')
    zip_framework = add('zip-link', 'PBXBuildFile', productRef=zip_product)
    groups = {}; sources = []; resources = []
    for folder in ['RoomLocate', 'Native']:
        refs = []
        for path in sorted((ROOT / folder).iterdir()):
            if not path.is_file():
                continue
            types = {'.swift': 'sourcecode.swift', '.mm': 'sourcecode.cpp.objcpp', '.cpp': 'sourcecode.cpp.cpp',
                     '.h': 'sourcecode.c.h', '.hpp': 'sourcecode.cpp.h', '.plist': 'text.plist.xml', '.txt': 'text'}
            ref = add('file:' + folder + '/' + path.name, 'PBXFileReference', lastKnownFileType=types[path.suffix], path=path.name, sourceTree='<group>')
            refs.append(ref)
            if path.suffix in ['.swift', '.mm', '.cpp']:
                fields = dict(fileRef=ref)
                if path.name == 'RoomBuilder.cpp':
                    fields['settings'] = dict(COMPILER_FLAGS='-O2')
                sources.append(add('compile:' + folder + '/' + path.name, 'PBXBuildFile', **fields))
            if path.suffix == '.txt':
                resources.append(add('resource:' + folder + '/' + path.name, 'PBXBuildFile', fileRef=ref))
        groups[folder] = add('group:' + folder, 'PBXGroup', path=folder, children=refs, sourceTree='<group>')
    products = add('products-group', 'PBXGroup', name='Products', children=[product], sourceTree='<group>')
    main_group = add('main-group', 'PBXGroup', children=[groups['RoomLocate'], groups['Native'], products], sourceTree='<group>')
    source_phase = add('source-phase', 'PBXSourcesBuildPhase', buildActionMask=2147483647, files=sources, runOnlyForDeploymentPostprocessing=0)
    framework_phase = add('framework-phase', 'PBXFrameworksBuildPhase', buildActionMask=2147483647, files=[framework, zip_framework], runOnlyForDeploymentPostprocessing=0)
    resource_phase = add('resource-phase', 'PBXResourcesBuildPhase', buildActionMask=2147483647, files=resources, runOnlyForDeploymentPostprocessing=0)
    project_configs, target_configs = [], []
    for configuration in ['Debug', 'Release']:
        project_settings = dict(CLANG_ENABLE_MODULES='YES', CLANG_ENABLE_OBJC_ARC='YES', CLANG_CXX_LANGUAGE_STANDARD='c++17',
                                GCC_C_LANGUAGE_STANDARD='gnu11', SDKROOT='iphoneos', IPHONEOS_DEPLOYMENT_TARGET='16.0')
        target_settings = dict(PRODUCT_NAME='$(TARGET_NAME)', PRODUCT_BUNDLE_IDENTIFIER='com.example.m202a.RoomLocate',
            INFOPLIST_FILE='RoomLocate/Info.plist', GENERATE_INFOPLIST_FILE='NO', CODE_SIGN_STYLE='Automatic', DEVELOPMENT_TEAM='',
            SWIFT_VERSION='5.0', SWIFT_OBJC_BRIDGING_HEADER='RoomLocate/RoomLocate-Bridging-Header.h',
            CLANG_CXX_LANGUAGE_STANDARD='c++17', TARGETED_DEVICE_FAMILY='1,2', SUPPORTED_PLATFORMS='iphoneos iphonesimulator',
            LD_RUNPATH_SEARCH_PATHS=['$(inherited)', '@executable_path/Frameworks'], ENABLE_USER_SCRIPT_SANDBOXING='YES',
            SWIFT_STRICT_CONCURRENCY='minimal', MARKETING_VERSION='1.1', CURRENT_PROJECT_VERSION='2')
        if configuration == 'Debug':
            project_settings.update(ONLY_ACTIVE_ARCH='YES', GCC_OPTIMIZATION_LEVEL='0', DEBUG_INFORMATION_FORMAT='dwarf', GCC_PREPROCESSOR_DEFINITIONS=['DEBUG=1', '$(inherited)'])
            target_settings.update(SWIFT_OPTIMIZATION_LEVEL='-Onone', SWIFT_ACTIVE_COMPILATION_CONDITIONS='DEBUG')
        else:
            project_settings.update(ONLY_ACTIVE_ARCH='NO', DEBUG_INFORMATION_FORMAT='dwarf-with-dsym')
            target_settings.update(SWIFT_OPTIMIZATION_LEVEL='-O', SWIFT_COMPILATION_MODE='wholemodule')
        project_configs.append(add('project:' + configuration, 'XCBuildConfiguration', name=configuration, buildSettings=project_settings))
        target_configs.append(add('target:' + configuration, 'XCBuildConfiguration', name=configuration, buildSettings=target_settings))
    pconfigs = add('project-configs', 'XCConfigurationList', buildConfigurations=project_configs, defaultConfigurationIsVisible=0, defaultConfigurationName='Release')
    tconfigs = add('target-configs', 'XCConfigurationList', buildConfigurations=target_configs, defaultConfigurationIsVisible=0, defaultConfigurationName='Release')
    target = add('app-target', 'PBXNativeTarget', name='RoomLocate', productName='RoomLocate', productReference=product,
                 productType='com.apple.product-type.application', buildConfigurationList=tconfigs,
                 buildPhases=[source_phase, framework_phase, resource_phase], buildRules=[], dependencies=[], packageProductDependencies=[opencv, zip_product])
    project = add('project', 'PBXProject', attributes=dict(BuildIndependentTargetsInParallel='YES', LastUpgradeCheck='1600'),
                  buildConfigurationList=pconfigs, compatibilityVersion='Xcode 14.0', developmentRegion='en', knownRegions=['en', 'zh-Hans', 'Base'],
                  mainGroup=main_group, productRefGroup=products, projectDirPath='', projectRoot='', targets=[target], packageReferences=[package, zip_package])
    path = ROOT / 'RoomLocate.xcodeproj'; path.mkdir(exist_ok=True)
    (path / 'project.pbxproj').write_text('// !$*UTF8*$!\n' + encode(dict(archiveVersion=1, classes={}, objectVersion=56, objects=objects, rootObject=project)) + '\n')
    schemes = path / 'xcshareddata/xcschemes'; schemes.mkdir(parents=True, exist_ok=True)
    reference = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="RoomLocate.app" BlueprintName="RoomLocate" ReferencedContainer="container:RoomLocate.xcodeproj"/>'
    (schemes / 'RoomLocate.xcscheme').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="1600" version="1.3">
  <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{reference}</BuildActionEntry></BuildActionEntries></BuildAction>
  <TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB"/>
  <LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></LaunchAction>
  <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></ProfileAction>
  <AnalyzeAction buildConfiguration="Debug"/>
  <ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>
''')
    print(path)


if __name__ == '__main__':
    main()
