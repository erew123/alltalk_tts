#!/bin/bash

# Get the conda environment path
env_path="${CONDA_DEFAULT_ENV}/lib"

# List of broken symlinks
declare -A symlinks=(
    ["libcufile.so"]="libcufile.so"
    ["libcufile_rdma.so"]="libcufile_rdma.so"
    ["libcudart.so"]="libcudart.so"
    ["libnvjpeg.so"]="libnvjpeg.so"
    ["libcurand.so"]="libcurand.so"
    ["libnvJitLink.so"]="libnvJitLink.so"
    ["libnvrtc-builtins.so"]="libnvrtc-builtins.so"
    ["libnvrtc.so"]="libnvrtc.so"
)

# Function to test for broken symlinks
test_broken_symlinks() {
    echo "Testing for broken symlinks..."
    broken_found=false
    for link in "${!symlinks[@]}"; do
        if [ -L "$env_path/$link" ] && [ ! -e "$env_path/$link" ]; then
            echo "Broken symlink found: $env_path/$link"
            broken_found=true
        fi
    done
    if [ "$broken_found" = false ]; then
        echo "No broken symlinks found."
    fi
}

# Function to fix broken symlinks
fix_broken_symlinks() {
    echo "Fixing broken symlinks..."
    for link in "${!symlinks[@]}"; do
        if [ -L "$env_path/$link" ] && [ ! -e "$env_path/$link" ]; then
            echo "Removing broken link: $env_path/$link"
            rm "$env_path/$link"

            # Find all possible target files
            target_files=($(find "$env_path" -name "${symlinks[$link]}*"))
            
            if [ ${#target_files[@]} -gt 0 ]; then
                echo "Select a target file for $link:"
                for i in "${!target_files[@]}"; do
                    echo "$((i+1)). ${target_files[$i]}"
                done

                read -p "Enter the number of the target file to use: " choice
                choice=$((choice-1))

                if [[ $choice -ge 0 && $choice -lt ${#target_files[@]} ]]; then
                    target="${target_files[$choice]}"
                    echo "Creating new symlink: $env_path/$link -> $target"
                    ln -s "$target" "$env_path/$link"
                else
                    echo "Invalid choice, skipping $link."
                fi
            else
                echo "No target files found for $link."
            fi
        fi
    done
    echo "Verification of symbolic links:"
    ls -l $env_path | grep -E "libcufile.so|libcudart.so|libcufile_rdma.so|libnvjpeg.so|libcurand.so|libnvJitLink.so|libnvrtc-builtins.so|libnvrtc.so"
}

# Menu
while true; do
    echo "Select an option:"
    echo "1. Test for broken symlinks"
    echo "2. Fix broken symlinks"
    echo "3. Exit"
    read -p "Enter your choice [1-3]: " choice
    case $choice in
        1)
            test_broken_symlinks
            ;;
        2)
            fix_broken_symlinks
            ;;
        3)
            echo "Exiting."
            break
            ;;
        *)
            echo "Invalid choice. Please select 1, 2, or 3."
            ;;
    esac
done

