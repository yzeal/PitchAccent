# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['pitch_accent_qt.py'],
    pathex=[],
    binaries=[
        ('ffmpeg.exe', '.'),  # Bundle ffmpeg.exe in the root directory
    ],
    datas=[],
    hiddenimports=[
        'numpy',
        'parselmouth',
        'sounddevice',
        'scipy',
        'cv2',
        'moviepy',
        'vlc',
        'pyqtgraph',
        'PIL',
        'matplotlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pitch_accent_qt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)