{
	description = "Rust Flake Exploration Mechanisms";

	inputs = {
		# better use commit
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
		devenv.url = "github:cachix/devenv";
		fenix.url = "github:nix-community/fenix";
        fenix.inputs = { nixpkgs.follows = "nixpkgs"; };
	};

	outputs = {self, nixpkgs, devenv, fenix} @ inputs:

        let system = "x86_64-linux";
        pkgs = nixpkgs.legacyPackages.${system};

		in {
		    packages.${system}.devenv-up = self.devShells.${system}.default.config.procfileScript;

	        devShells.${system}.default = devenv.lib.mkShell {
	            inherit inputs pkgs;
	            modules = [
	                ({ pkgs, config, ... }: {
	                    packages = with pkgs; [llvmPackages.libclang.lib futhark clang];
	                    env.LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
	                    languages.python.enable = true;
                    # This is your devenv configuration
                        languages.rust = {
                        enable = true;
                        # https://devenv.sh/reference/options/#languagesrustchannel
                        channel = "stable";
                        components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
                        };

                        enterShell = ''
                           echo Rust shell entered
                        '';
                    })
                ];
            };
	    };
}