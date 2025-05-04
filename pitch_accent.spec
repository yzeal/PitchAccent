# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['pitch_accent_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.io.matlab.streams',
        'scipy.sparse.csgraph._validation',
        'scipy.sparse._csparsetools',
        'scipy.special._ufuncs_cxx',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.integrate',
        'scipy.integrate.quadrature',
        'scipy.integrate.odepack',
        'scipy.integrate._odepack',
        'scipy.integrate.quadpack',
        'scipy.integrate._quadpack',
        'scipy.integrate._ode',
        'scipy.integrate.vode',
        'scipy.integrate._dop',
        'scipy.integrate.lsoda',
        'moviepy',
        'moviepy.audio.fx.all',
        'moviepy.video.fx.all',
        'tkinterdnd2'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PitchAccentTrainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)