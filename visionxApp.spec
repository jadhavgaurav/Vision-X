
a = Analysis(
    ['visionxApp.py'],
    pathex=[],
    binaries=[],
    # Add the src folder to the path so it can find our visionx module
    datas=[
        ('src/visionx', 'src/visionx'),
        ('models', 'models'),
        ('resources', 'resources'),
        ('database', 'database')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VisionX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # Set to False to hide the console window in the final app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)