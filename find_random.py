def find_global_random_usage():
    """Find where global random is being used"""
    import subprocess

    # Find files using global random
    print("🔍 Searching for global random usage...")

    # Command to find random. calls (excluding self.rng.)
    cmd = r'grep -r "random\." finchge/ --include="*.py" | grep -v "self\.rng" | grep -v "\.rng\."'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print("Found these uses of global random:")
        print(result.stdout)

        # Also check for np.random
        cmd_np = r'grep -r "np\.random\." finchge/ --include="*.py" | grep -v "self\.np_rng" | grep -v "\.np_rng\."'
        result_np = subprocess.run(cmd_np, shell=True, capture_output=True, text=True)

        if result_np.stdout:
            print("\nFound these uses of global np.random:")
            print(result_np.stdout)
    else:
        print("✅ No global random usage found (grep says)")


if __name__ == "__main__":
    find_global_random_usage()
