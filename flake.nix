{
  description = "SatQuery AI - agentic vision-language assistant for remote-sensing imagery";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # The project installs Python wheels from PyPI. A manylinux wheel expects
        # libstdc++, zlib, expat and friends on the standard library path, which
        # NixOS does not provide - so they go on LD_LIBRARY_PATH instead. That is
        # far less work than packaging rasterio and GDAL through nixpkgs, and it
        # keeps requirements.txt as the single source of truth for versions.
        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib   # libstdc++ - numpy, rasterio, pillow
          zlib
          expat              # GDAL XML parsing
          libxml2
          libxcrypt-legacy
          sqlite             # the PROJ database
          glib
          zstd
          bzip2
          xz
          openssl
          curl
          libjpeg
          libtiff
          libwebp
          libdeflate
          libGL              # pillow and, in Phase D, opencv
        ];

        python = pkgs.python313;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.gnumake ] ++ runtimeLibs;

          env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

          shellHook = ''
            export SATQUERY_ROOT="$PWD"
            if [ -d .venv ]; then
              source .venv/bin/activate
              echo "SatQuery AI dev shell - $(python --version 2>&1), .venv active"
            else
              echo "SatQuery AI dev shell - $(python3 --version 2>&1)"
              echo "no .venv yet: run ./scripts/setup.sh"
            fi
          '';
        };
      });
}
